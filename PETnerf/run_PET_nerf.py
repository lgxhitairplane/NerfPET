import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time

import numpy as np
import tensorflow as tf

try:
    from PETnerf.PETScanner import PETScanner
    from PETnerf.config import get_scanner_config
except ImportError:
    from PETScanner import PETScanner
    from config import get_scanner_config


tf.compat.v1.enable_eager_execution()


def poisson_nll(pred, target, eps=1e-8):
    """Poisson negative log likelihood without the target-only log(y!) term."""
    pred = tf.maximum(pred, eps)
    return tf.reduce_mean(pred - target * tf.math.log(pred))


def sample_pdf(bins, weights, N_samples, det=False):
    weights += 1e-5
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds - 1)
    above = tf.minimum(cdf.shape[-1] - 1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


def get_bin_cut_range(bin_num, bin_cut_ratio):
    if bin_cut_ratio < 0. or bin_cut_ratio >= 0.5:
        raise ValueError('bin_cut_ratio must be in [0, 0.5).')
    if bin_cut_ratio == 0.:
        return 0, bin_num
    bin_start = int(np.floor(bin_cut_ratio * bin_num)) + 1
    bin_stop = int(np.ceil((1. - bin_cut_ratio) * bin_num))
    if bin_stop <= bin_start:
        raise ValueError('bin_cut_ratio leaves no valid bins.')
    return bin_start, bin_stop


def sample_lor_indices(total, batch_size, allow_repeated=False,
                       bin_num=None, view_num=None, bin_start=None,
                       bin_stop=None, sample_mode='global'):
    if bin_num is not None:
        valid_bin_num = bin_stop - bin_start
        slice_num = total // (view_num * bin_num)
        valid_total = slice_num * view_num * valid_bin_num
        total = valid_total

    if batch_size > total:
        raise ValueError('N_rand cannot be larger than the number of LORs.')

    if bin_num is None:
        if allow_repeated:
            return np.random.randint(0, total, size=batch_size, dtype=np.int64)

        selected = set()
        while len(selected) < batch_size:
            need = batch_size - len(selected)
            draw = np.random.randint(0, total, size=need * 2)
            selected.update(int(x) for x in draw)
        return np.fromiter(selected, dtype=np.int64, count=batch_size)

    if sample_mode == 'slice':
        per_slice_total = view_num * valid_bin_num
        if batch_size > per_slice_total and not allow_repeated:
            raise ValueError(
                'N_rand cannot be larger than the valid LORs in one slice '
                'when --lor_sample_mode slice and repeated LORs are disabled.')
        slice_id = np.random.randint(0, slice_num, dtype=np.int64)
        if allow_repeated:
            ray_offsets = np.random.randint(
                0, per_slice_total, size=batch_size, dtype=np.int64)
        else:
            ray_offsets = np.random.choice(
                per_slice_total, size=batch_size, replace=False)
        views = ray_offsets // valid_bin_num
        bins = bin_start + ray_offsets % valid_bin_num
        return slice_id * view_num * bin_num + views * bin_num + bins

    if sample_mode != 'global':
        raise ValueError('Unknown LOR sample mode: {}'.format(sample_mode))

    if allow_repeated:
        slice_ids = np.random.randint(
            0, slice_num,
            size=batch_size,
            dtype=np.int64)
        views = np.random.randint(0, view_num, size=batch_size, dtype=np.int64)
        bins = np.random.randint(
            bin_start, bin_stop, size=batch_size, dtype=np.int64)
        return (slice_ids * view_num * bin_num + views * bin_num + bins)

    selected = set()
    while len(selected) < batch_size:
        need = batch_size - len(selected)
        slice_ids = np.random.randint(0, slice_num, size=need * 2)
        views = np.random.randint(0, view_num, size=need * 2)
        bins = np.random.randint(bin_start, bin_stop, size=need * 2)
        draw = slice_ids * view_num * bin_num + views * bin_num + bins
        selected.update(int(x) for x in draw)
    return np.fromiter(selected, dtype=np.int64, count=batch_size)


class Embedder:
    """Sin/cos positional encoding for an arbitrary input dimension."""

    def __init__(self, input_dims, multires, include_input=True):
        self.input_dims = input_dims
        self.multires = multires
        self.include_input = include_input
        self.embed_fns = []
        self.out_dim = 0
        self.create_embedding_fn()

    def create_embedding_fn(self):
        if self.include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dim += self.input_dims

        freq_bands = 2. ** tf.linspace(0., self.multires - 1, self.multires)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: tf.math.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: tf.math.cos(x * freq))
            self.out_dim += 2 * self.input_dims

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, multires, i=0):
    if i == -1:
        return tf.identity, input_dims

    embedder_obj = Embedder(input_dims=input_dims, multires=multires)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


def init_pet_nerf_model(D=8, W=256, input_ch_pts=3, output_ch=2,
                        skips=(4,)):
    """MLP for PET-NeRF: (x, y, z) -> (intensity, density)."""

    relu = tf.keras.layers.ReLU()

    def dense(width, act=relu):
        return tf.keras.layers.Dense(width, activation=act)

    inputs = tf.keras.Input(shape=(input_ch_pts,))
    inputs_pts = tf.keras.layers.Lambda(
        lambda x: x[..., :input_ch_pts],
        output_shape=(input_ch_pts,))(inputs)

    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.keras.layers.Concatenate(axis=-1)(
                [inputs_pts, outputs])

    bottleneck = dense(W, act=None)(outputs)
    outputs = dense(W // 2)(bottleneck)
    outputs = dense(output_ch, act=None)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def angle_to_dir(theta, fai):
    """Convert spherical angles to a unit LOR direction."""
    return tf.stack([
        tf.sin(theta) * tf.cos(fai),
        tf.sin(theta) * tf.sin(fai),
        tf.cos(theta),
    ], axis=-1)


def dir_to_angle(lor_dirs):
    """Convert unit LOR directions to theta/fai spherical angles."""
    z = tf.clip_by_value(lor_dirs[..., 2], -1., 1.)
    theta = tf.acos(z)
    fai = tf.atan2(lor_dirs[..., 1], lor_dirs[..., 0])
    return tf.stack([theta, fai], axis=-1)


def intersect_lor_aabb_tf(starts, dirs, lengths, bounds):
    """Return clipped near/far distances where finite LOR segments hit an AABB."""
    bounds = tf.convert_to_tensor(bounds, dtype=starts.dtype)
    box_min = tf.stack([bounds[0], bounds[2], bounds[4]])
    box_max = tf.stack([bounds[1], bounds[3], bounds[5]])
    safe_dirs = tf.where(
        tf.abs(dirs) < 1e-8,
        tf.ones_like(dirs) * 1e-8,
        dirs)
    inv_d = 1. / safe_dirs
    t0 = (box_min - starts) * inv_d
    t1 = (box_max - starts) * inv_d
    tmin = tf.minimum(t0, t1)
    tmax = tf.maximum(t0, t1)
    near = tf.maximum(tf.reduce_max(tmin, axis=-1, keepdims=True), 0.)
    far = tf.minimum(tf.reduce_min(tmax, axis=-1, keepdims=True), lengths)
    hit = far > near
    near = tf.where(hit, near, tf.zeros_like(near))
    far = tf.where(hit, far, tf.zeros_like(far))
    return near, far, hit


def run_pet_network(points, angles, fn, embed_pts_fn, netchunk=1024 * 64):
    """Encode sampled points, then query the PET-NeRF MLP.

    PET activity and attenuation are modeled as spatial scalar fields, so the
    LOR/view direction is intentionally ignored.
    """

    points_flat = tf.reshape(points, [-1, points.shape[-1]])
    embedded = embed_pts_fn(points_flat)

    outputs_flat = tf.concat(
        [fn(embedded[i:i + netchunk])
         for i in range(0, embedded.shape[0], netchunk)],
        axis=0)
    outputs = tf.reshape(
        outputs_flat,
        list(points.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def activate_pet_raw(raw, raw_activation='softplus', density_max=0.015):
    """Convert raw network outputs to physical PET field values.

    Activity/intensity is non-negative. Attenuation density (mu) is bounded to
    [0, density_max].
    """
    if raw_activation == 'relu':
        intensity = tf.nn.relu(raw[..., 0])
    elif raw_activation == 'softplus':
        intensity = tf.nn.softplus(raw[..., 0])
    else:
        raise ValueError('Unknown PET raw activation: {}'.format(raw_activation))
    if raw.shape[-1] == 1:
        density = tf.zeros_like(intensity)
    else:
        density = tf.math.sigmoid(raw[..., 1]) * density_max
    return intensity, density


def raw2pet_outputs(raw, z_vals, raw_activation='softplus',
                    density_max=0.015, density_override=None,
                    activity_mask=None):
    """Project PET-NeRF raw predictions along each LOR.

    The network predicts PET activity c(x) and attenuation density mu(x).
    A coincidence event can originate anywhere along the LOR, while attenuation
    affects the full path. Therefore the expected count is the activity line
    integral multiplied by the total LOR transmission:

      T_LOR = exp(-sum_i mu_i * delta_i)
      y = T_LOR * sum_i c_i * delta_i
    """

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = tf.concat([dists, dists[..., -1:]], axis=-1)

    intensity, density = activate_pet_raw(
        raw,
        raw_activation=raw_activation,
        density_max=density_max)
    if density_override is not None:
        density = tf.maximum(density_override, 0.)
    if activity_mask is not None:
        intensity = intensity * activity_mask

    alpha = 1. - tf.exp(-density * dists)
    density_integral = tf.reduce_sum(density * dists, axis=-1)
    transmission = tf.exp(-density_integral)
    activity_integral = tf.reduce_sum(intensity * dists, axis=-1)
    weights = transmission[..., None] * intensity * dists
    count = transmission * activity_integral

    return {
        'count': count[..., None],
        'intensity_alpha_sum': activity_integral,
        'density_integral': density_integral,
        'transmission': transmission,
        'alpha': alpha,
        'weights': weights,
        'intensity': intensity,
        'density': density,
        'dists': dists,
        # Backward-compatible aliases for existing experiment scripts.
        'activity_integral': activity_integral,
        'mu_integral': density_integral,
        'activity': intensity,
        'mu': density,
    }


def load_attn_map(path, shape=None, dtype='float32', header_bytes=0):
    """Load a fixed attenuation map used as mu instead of a learned field."""
    dtype = np.dtype(dtype)

    if path.endswith('.npy'):
        attn = np.load(path).astype(np.float32)
    elif path.endswith('.npz'):
        data = np.load(path)
        key = 'mu' if 'mu' in data else 'attn'
        if key not in data:
            key = 'density' if 'density' in data else data.files[0]
        attn = data[key].astype(np.float32)
    else:
        if shape is None:
            raise ValueError('--attn_shape is required for raw attenuation maps.')
        shape = tuple(int(x) for x in shape)
        expected_values = int(np.prod(shape))
        payload_bytes = os.path.getsize(path) - header_bytes
        expected_bytes = expected_values * dtype.itemsize
        if payload_bytes != expected_bytes:
            raise ValueError(
                '{} contains {} payload bytes, but shape {} with dtype {} '
                'expects {} bytes.'.format(
                    path, payload_bytes, shape, dtype, expected_bytes))
        attn = np.memmap(
            path,
            dtype=dtype,
            mode='r',
            offset=header_bytes,
            shape=shape,
        ).astype(np.float32)

    if shape is not None:
        shape = tuple(int(x) for x in shape)
        if attn.size != int(np.prod(shape)):
            raise ValueError(
                'Attenuation map has {} values, but shape {} expects {}.'.format(
                    attn.size, shape, int(np.prod(shape))))
        attn = attn.reshape(shape)

    attn = np.squeeze(attn).astype(np.float32)
    if attn.ndim != 2:
        raise ValueError(
            'Only 2D attenuation maps are currently supported, got shape {}.'.format(
                attn.shape))
    return attn


def query_attn_map_2d(points, attn_map, bounds, outside_value=0.):
    """Bilinearly query mu from a 2D attenuation image at sampled LOR points."""
    xmin, xmax, ymin, ymax = bounds
    height = tf.shape(attn_map)[0]
    width = tf.shape(attn_map)[1]
    x = points[..., 0]
    y = points[..., 1]

    gx = (x - xmin) / tf.maximum(xmax - xmin, 1e-8) * tf.cast(width - 1, tf.float32)
    gy = (y - ymin) / tf.maximum(ymax - ymin, 1e-8) * tf.cast(height - 1, tf.float32)
    inside = tf.logical_and(
        tf.logical_and(gx >= 0., gx <= tf.cast(width - 1, tf.float32)),
        tf.logical_and(gy >= 0., gy <= tf.cast(height - 1, tf.float32)))

    gx = tf.clip_by_value(gx, 0., tf.cast(width - 1, tf.float32))
    gy = tf.clip_by_value(gy, 0., tf.cast(height - 1, tf.float32))
    x0 = tf.cast(tf.floor(gx), tf.int32)
    y0 = tf.cast(tf.floor(gy), tf.int32)
    x1 = tf.minimum(x0 + 1, width - 1)
    y1 = tf.minimum(y0 + 1, height - 1)

    wx = gx - tf.cast(x0, tf.float32)
    wy = gy - tf.cast(y0, tf.float32)
    v00 = tf.gather_nd(attn_map, tf.stack([y0, x0], axis=-1))
    v10 = tf.gather_nd(attn_map, tf.stack([y0, x1], axis=-1))
    v01 = tf.gather_nd(attn_map, tf.stack([y1, x0], axis=-1))
    v11 = tf.gather_nd(attn_map, tf.stack([y1, x1], axis=-1))
    values = (
        v00 * (1. - wx) * (1. - wy) +
        v10 * wx * (1. - wy) +
        v01 * (1. - wx) * wy +
        v11 * wx * wy)
    return tf.where(inside, values, tf.ones_like(values) * outside_value)


def soft_activity_mask_from_mu(mu, threshold=1e-5, softness=1e-4):
    softness = tf.maximum(tf.cast(softness, mu.dtype), tf.cast(1e-12, mu.dtype))
    threshold = tf.cast(threshold, mu.dtype)
    return tf.sigmoid((mu - threshold) / softness)


def mask_fov_outputs(outputs, fov_hit_flat, keys):
    for key in keys:
        condition = fov_hit_flat
        if len(outputs[key].shape) > 1:
            condition = fov_hit_flat[..., None]
            condition = tf.broadcast_to(condition, tf.shape(outputs[key]))
        outputs[key] = tf.where(
            condition,
            outputs[key],
            tf.zeros_like(outputs[key]))


def render_lors(lor_batch, network_fn, network_query_fn, N_samples,
                perturb=0., N_importance=0, network_fine=None,
                raw_activation='softplus', density_max=0.015,
                fov_bounds=None, attn_map=None, attn_bounds=None,
                attn_activity_soft_mask=False):
    """Render a batch of PET LORs.

    lor_batch shape: [N_lors, 6]
      columns 0:3 = crystal 1 position, the start point of the LOR
      columns 3:6 = crystal 2 position, the end point of the LOR
    """

    crystal1 = lor_batch[:, :3]
    crystal2 = lor_batch[:, 3:6]
    lor_vec = crystal2 - crystal1
    lor_length = tf.linalg.norm(lor_vec, axis=-1, keepdims=True)
    lor_dirs = lor_vec / tf.maximum(lor_length, 1e-8)
    if fov_bounds is None:
        z_near = tf.zeros_like(lor_length)
        z_far = lor_length
        fov_hit = tf.ones_like(lor_length, dtype=tf.bool)
    else:
        z_near, z_far, fov_hit = intersect_lor_aabb_tf(
            crystal1, lor_dirs, lor_length, fov_bounds)

    t_vals = tf.linspace(0., 1., N_samples)
    z_vals = z_near * (1. - t_vals) + z_far * t_vals
    z_vals = tf.broadcast_to(z_vals, [lor_batch.shape[0], N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        z_vals = lower + (upper - lower) * tf.random.uniform(z_vals.shape)

    points = crystal1[:, None, :] + lor_dirs[:, None, :] * z_vals[..., None]
    raw = network_query_fn(points, None, network_fn)
    density_override = None
    activity_mask = None
    if attn_map is not None:
        density_override = query_attn_map_2d(points, attn_map, attn_bounds)
        if attn_activity_soft_mask:
            activity_mask = soft_activity_mask_from_mu(density_override)
    outputs = raw2pet_outputs(
        raw,
        z_vals,
        raw_activation=raw_activation,
        density_max=density_max,
        density_override=density_override,
        activity_mask=activity_mask)
    fov_hit_flat = tf.squeeze(fov_hit, axis=-1)
    mask_fov_outputs(
        outputs,
        fov_hit_flat,
        ['count', 'intensity_alpha_sum', 'density_integral',
         'transmission', 'alpha', 'weights', 'intensity', 'density',
         'dists', 'activity_integral', 'mu_integral',
         'activity', 'mu'])

    outputs.update({
        'raw': raw,
        'z_vals': z_vals,
        'points': points,
        'fov_hit': fov_hit,
        'fov_near': z_near,
        'fov_far': z_far,
    })

    if N_importance > 0:
        count0 = outputs['count']
        intensity_alpha_sum0 = outputs['intensity_alpha_sum']
        density_integral0 = outputs['density_integral']

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            outputs['weights'][..., 1:-1],
            N_importance,
            det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        points = crystal1[:, None, :] + lor_dirs[:, None, :] * z_vals[..., None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(points, None, run_fn)
        density_override = None
        activity_mask = None
        if attn_map is not None:
            density_override = query_attn_map_2d(points, attn_map, attn_bounds)
            if attn_activity_soft_mask:
                activity_mask = soft_activity_mask_from_mu(density_override)
        outputs = raw2pet_outputs(
            raw,
            z_vals,
            raw_activation=raw_activation,
            density_max=density_max,
            density_override=density_override,
            activity_mask=activity_mask)
        mask_fov_outputs(
            outputs,
            fov_hit_flat,
            ['count', 'intensity_alpha_sum', 'density_integral',
             'transmission', 'alpha', 'weights', 'intensity', 'density',
             'dists', 'activity_integral', 'mu_integral',
             'activity', 'mu'])
        outputs.update({
            'raw': raw,
            'z_vals': z_vals,
            'points': points,
            'count0': count0,
            'intensity_alpha_sum0': intensity_alpha_sum0,
            'density_integral0': density_integral0,
            'z_std': tf.math.reduce_std(z_samples, -1),
            'fov_hit': fov_hit,
            'fov_near': z_near,
            'fov_far': z_far,
        })

    return outputs


def create_pet_nerf(args):
    embed_pts_fn, input_ch_pts = get_embedder(3, args.multires, args.i_embed)
    use_attn_map = getattr(args, 'attn_image', None) is not None
    output_ch = 1 if use_attn_map else 2

    model = init_pet_nerf_model(
        D=args.netdepth,
        W=args.netwidth,
        input_ch_pts=input_ch_pts,
        output_ch=output_ch,
        skips=args.net_skips_parsed,
    )
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_pet_nerf_model(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch_pts=input_ch_pts,
            output_ch=output_ch,
            skips=args.net_skips_parsed,
        )
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(points, angles, network_fn):
        return run_pet_network(
            points, angles, network_fn,
            embed_pts_fn=embed_pts_fn,
            netchunk=args.netchunk)

    render_kwargs_train = {
        'network_fn': model,
        'network_query_fn': network_query_fn,
        'N_samples': args.N_samples,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'raw_activation': args.raw_activation,
        'density_max': args.density_max,
        'fov_bounds': getattr(args, 'fov_bounds_parsed', None),
    }
    if use_attn_map:
        attn_map = load_attn_map(
            args.attn_image,
            shape=getattr(args, 'attn_shape_parsed', None),
            dtype=args.attn_dtype,
            header_bytes=args.attn_header_bytes)
        render_kwargs_train['attn_map'] = tf.convert_to_tensor(
            attn_map, dtype=tf.float32)
        render_kwargs_train['attn_bounds'] = [
            float(x) for x in args.attn_bounds_parsed]
        render_kwargs_train['attn_activity_soft_mask'] = (
            args.attn_activity_soft_mask)
        print('Using fixed attenuation map')
        print('  attn_image', args.attn_image)
        print('  attn_shape', attn_map.shape)
        print('  attn_range', float(attn_map.min()), float(attn_map.max()))
        print('  attn_bounds', args.attn_bounds_parsed)
        print('  attn_activity_soft_mask', args.attn_activity_soft_mask)
    render_kwargs_test = dict(render_kwargs_train)
    render_kwargs_test['perturb'] = 0.

    return render_kwargs_train, render_kwargs_test, grad_vars, models


def estimate_n_samples_from_scanner(scanner, sample_step):
    """Estimate a global sample count so every LOR is sampled at <= sample_step."""
    centers = scanner.get_all_crystal_centers()
    xyz_min = centers.min(axis=0)
    xyz_max = centers.max(axis=0)
    max_lor_length = np.linalg.norm(xyz_max - xyz_min)
    return int(np.ceil(max_lor_length / sample_step)) + 1, max_lor_length


def estimate_n_samples_from_bounds(bounds, sample_step):
    """Estimate sample count from the longest possible segment inside an AABB."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    diagonal = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    return int(np.ceil(diagonal / sample_step)) + 1, float(diagonal)


def load_pet_npz(datadir):
    """Load PET LOR data from an npz file.

    Expected arrays:
      crystal1: [N, 3], detector crystal 1 positions
      crystal2: [N, 3], detector crystal 2 positions
      targets:     [N] or [N, 1], measured LOR count/projection value
    """

    data = np.load(datadir)
    crystal1 = data['crystal1'].astype(np.float32)
    crystal2 = data['crystal2'].astype(np.float32)
    targets = data['targets'].astype(np.float32)
    targets = targets.reshape([-1, 1])

    lors = np.concatenate([crystal1, crystal2], axis=-1)
    return lors.astype(np.float32), targets.astype(np.float32)


def load_mich_counts(datadir, scanner, dtype='float32', header_bytes=0):
    """Load .mich data whose element i stores the measured count of LORID i."""
    lor_num = scanner.get_lor_num()
    dtype = np.dtype(dtype)
    payload_bytes = os.path.getsize(datadir) - header_bytes
    if payload_bytes < 0:
        raise ValueError('mich_header_bytes is larger than the input file.')
    itemsize = dtype.itemsize
    if payload_bytes % itemsize != 0:
        raise ValueError(
            'The .mich payload size is not divisible by dtype size. '
            'Check --mich_dtype or --mich_header_bytes.')
    file_lor_num = payload_bytes // itemsize
    if file_lor_num != lor_num:
        raise ValueError(
            'The .mich file contains {} values, but scanner {} expects {} LORs. '
            'Check --scanner, --mich_dtype, or --mich_header_bytes.'.format(
                file_lor_num, scanner.__class__.__name__, lor_num))
    targets = np.memmap(
        datadir,
        dtype=dtype,
        mode='r',
        offset=header_bytes,
        shape=(lor_num,),
    )
    return targets


def make_demo_lors(N=4096):
    """Tiny synthetic LOR set so the framework can run before real data exists."""

    mids = np.random.uniform(-0.4, 0.4, size=(N, 3)).astype(np.float32)
    theta_fai = np.stack([
        np.random.uniform(0., np.pi, size=N),
        np.random.uniform(-np.pi, np.pi, size=N),
    ], axis=-1).astype(np.float32)
    dirs = np.stack([
        np.sin(theta_fai[:, 0]) * np.cos(theta_fai[:, 1]),
        np.sin(theta_fai[:, 0]) * np.sin(theta_fai[:, 1]),
        np.cos(theta_fai[:, 0]),
    ], axis=-1).astype(np.float32)
    half_length = 1.5
    crystal1 = mids - half_length * dirs
    crystal2 = mids + half_length * dirs

    radius = np.linalg.norm(mids, axis=-1, keepdims=True)
    targets = np.exp(-4. * radius * radius).astype(np.float32)

    lors = np.concatenate([crystal1, crystal2], axis=-1)
    return lors, targets


def parse_int3(value):
    parts = [int(x) for x in str(value).split(',')]
    if len(parts) == 1:
        return parts * 3
    if len(parts) != 3:
        raise ValueError('Expected one int or three comma-separated ints.')
    return parts


def parse_int2_or_3(value):
    parts = [int(x) for x in str(value).split(',')]
    if len(parts) not in (2, 3):
        raise ValueError('Expected two or three comma-separated ints.')
    return parts


def parse_int_list(value):
    if value is None or str(value).strip() == '':
        return []
    return [int(x) for x in str(value).split(',') if str(x).strip() != '']


def parse_float3(value):
    parts = [float(x) for x in str(value).split(',')]
    if len(parts) == 1:
        return parts * 3
    if len(parts) != 3:
        raise ValueError('Expected one float or three comma-separated floats.')
    return parts


def parse_float4(value):
    parts = [float(x) for x in str(value).split(',')]
    if len(parts) != 4:
        raise ValueError('Expected bounds as xmin,xmax,ymin,ymax.')
    return parts


def parse_float6(value):
    parts = [float(x) for x in str(value).split(',')]
    if len(parts) != 6:
        raise ValueError(
            'Expected bounds as xmin,xmax,ymin,ymax,zmin,zmax.')
    return parts


def get_fov_bounds(args):
    if args.fov_bounds is not None:
        bounds = parse_float6(args.fov_bounds)
    elif args.fov_size is not None:
        center = parse_float3(args.fov_center)
        size = parse_float3(args.fov_size)
        if any(s <= 0. for s in size):
            raise ValueError('--fov_size values must be positive.')
        bounds = [
            center[0] - size[0] * 0.5,
            center[0] + size[0] * 0.5,
            center[1] - size[1] * 0.5,
            center[1] + size[1] * 0.5,
            center[2] - size[2] * 0.5,
            center[2] + size[2] * 0.5,
        ]
    else:
        return None

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        raise ValueError('FOV bounds must satisfy min < max on every axis.')
    return [float(x) for x in bounds]


def get_attn_bounds(args, scanner):
    if args.attn_image is None:
        return None
    if args.attn_bounds is not None:
        bounds = parse_float4(args.attn_bounds)
    else:
        radius = scanner.get_radius()
        bounds = [-radius, radius, -radius, radius]
    xmin, xmax, ymin, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError('Attenuation bounds must satisfy min < max on x/y.')
    return [float(x) for x in bounds]


def parse_int2(value):
    parts = [int(x) for x in str(value).split(',')]
    if len(parts) == 1:
        return parts * 2
    if len(parts) != 2:
        raise ValueError('Expected one int or two comma-separated ints.')
    return parts


def get_default_volume_bounds(scanner, margin=0.):
    centers = scanner.get_all_crystal_centers()
    radius = scanner.get_radius()
    zmin = float(np.min(centers[:, 2])) - margin
    zmax = float(np.max(centers[:, 2])) + margin
    return [-radius - margin, radius + margin,
            -radius - margin, radius + margin,
            zmin, zmax]


def get_volume_directions(mode):
    if mode == 'z':
        dirs = [[0., 0., 1.]]
    elif mode == 'xyz':
        dirs = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    elif mode == 'six':
        dirs = [
            [1., 0., 0.], [-1., 0., 0.],
            [0., 1., 0.], [0., -1., 0.],
            [0., 0., 1.], [0., 0., -1.],
        ]
    else:
        raise ValueError('Unknown volume direction mode: {}'.format(mode))
    return np.asarray(dirs, dtype=np.float32)


def query_pet_field(points, directions, model, network_query_fn, chunk,
                    raw_activation='softplus', density_max=0.015,
                    attn_map=None, attn_bounds=None,
                    attn_activity_soft_mask=False):
    """Query intensity/density on flat 3D points."""
    intensity_parts = []
    density_parts = []
    for i in range(0, points.shape[0], chunk):
        pts = tf.convert_to_tensor(points[i:i + chunk][None, ...],
                                   dtype=tf.float32)
        raw = network_query_fn(pts, None, model)[0]
        intensity, density = activate_pet_raw(
            raw,
            raw_activation=raw_activation,
            density_max=density_max)
        if attn_map is not None:
            density = query_attn_map_2d(
                pts,
                attn_map,
                attn_bounds)[0]
            if attn_activity_soft_mask:
                intensity = intensity * soft_activity_mask_from_mu(density)
        intensity_parts.append(intensity.numpy())
        density_parts.append(density.numpy())

    return (
        np.concatenate(intensity_parts, axis=0),
        np.concatenate(density_parts, axis=0),
    )


def load_render_model(weights_path, models):
    """Load a saved PET-NeRF model, preferring fine weights when available."""
    basename = os.path.basename(weights_path)
    if basename.startswith('model_fine_'):
        if 'model_fine' not in models:
            raise ValueError(
                '--N_importance must be > 0 when loading model_fine weights.')
        model = models['model_fine']
        try:
            model.set_weights(list(np.load(weights_path, allow_pickle=True)))
        except ValueError as exc:
            raise ValueError(
                'Fine weights are incompatible with the current PET-NeRF '
                'architecture. Retrain after changing angle conditioning or '
                'density activation, or load a matching coarse model instead. '
                'Original error: {}'.format(exc))
        return model

    model = models['model']
    try:
        model.set_weights(list(np.load(weights_path, allow_pickle=True)))
    except ValueError as exc:
        raise ValueError(
            'Weights are incompatible with the current PET-NeRF architecture. '
            'Retrain after changing angle conditioning or density activation. '
            'Original error: {}'.format(exc))

    if 'model_fine' in models and basename.startswith('model_'):
        fine_path = os.path.join(
            os.path.dirname(weights_path),
            basename.replace('model_', 'model_fine_', 1))
        if os.path.exists(fine_path):
            fine_model = models['model_fine']
            try:
                fine_model.set_weights(list(np.load(fine_path, allow_pickle=True)))
            except ValueError as exc:
                print('skipping incompatible fine weights at', fine_path)
                print('  using coarse weights instead:', weights_path)
                print('  fine load error:', exc)
            else:
                print('loaded fine weights at', fine_path)
                return fine_model

    return model


def enable_fine_model_if_saved(args):
    if args.N_importance > 0 or args.ft_weights is None:
        return

    basename = os.path.basename(args.ft_weights)
    fine_path = None
    if basename.startswith('model_fine_'):
        fine_path = args.ft_weights
    elif basename.startswith('model_'):
        fine_path = os.path.join(
            os.path.dirname(args.ft_weights),
            basename.replace('model_', 'model_fine_', 1))

    if fine_path is not None and os.path.exists(fine_path):
        args.N_importance = 1


def export_pet_volume(args):
    """Sample a trained PET-NeRF into explicit 3D intensity/density volumes."""
    if args.ft_weights is None:
        raise ValueError('--ft_weights is required when exporting a volume.')

    scanner = PETScanner(get_scanner_config(args.scanner))
    enable_fine_model_if_saved(args)
    render_kwargs_train, _, _, models = create_pet_nerf(args)
    model = load_render_model(args.ft_weights, models)

    resolution = parse_int3(args.volume_resolution)
    if args.volume_bounds is None:
        bounds = get_default_volume_bounds(scanner, args.volume_margin)
    else:
        bounds = parse_float6(args.volume_bounds)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = resolution
    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    zs = np.linspace(zmin, zmax, nz, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    points = grid.reshape([-1, 3]).astype(np.float32)
    directions = get_volume_directions(args.volume_direction_mode)

    print('Exporting PET volume')
    print('  weights:', args.ft_weights)
    print('  resolution:', resolution)
    print('  bounds:', bounds)
    print('  direction_mode:', args.volume_direction_mode)

    intensity, density = query_pet_field(
        points,
        directions,
        model,
        render_kwargs_train['network_query_fn'],
        args.volume_chunk,
        raw_activation=args.raw_activation,
        density_max=args.density_max,
        attn_map=render_kwargs_train.get('attn_map'),
        attn_bounds=render_kwargs_train.get('attn_bounds'),
        attn_activity_soft_mask=render_kwargs_train.get(
            'attn_activity_soft_mask', False))
    intensity = intensity.reshape([nx, ny, nz])
    density = density.reshape([nx, ny, nz])

    output = args.volume_output
    if output is None:
        output = os.path.join(
            args.basedir, args.expname, 'reconstruction_volume.npz')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(
        output,
        intensity=intensity,
        density=density,
        # Backward-compatible aliases for existing analysis scripts.
        activity=intensity,
        mu=density,
        x=xs,
        y=ys,
        z=zs,
        bounds=np.asarray(bounds, dtype=np.float32),
        resolution=np.asarray(resolution, dtype=np.int32),
        direction_mode=np.asarray(args.volume_direction_mode),
    )
    print('saved volume at', output)
    print('intensity range', float(intensity.min()), float(intensity.max()))
    print('density range', float(density.min()), float(density.max()))
    return output


def normalize(v, eps=1e-8):
    return v / max(float(np.linalg.norm(v)), eps)


def make_camera_rays(H, W, focal, eye, target=np.zeros(3, dtype=np.float32)):
    """Generate pinhole camera rays for a camera looking at target."""
    forward = normalize(target - eye)
    world_up = np.asarray([0., 0., 1.], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.asarray([0., 1., 0.], dtype=np.float32)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    dirs = (
        ((i - W * 0.5) / focal)[..., None] * right +
        (-(j - H * 0.5) / focal)[..., None] * up +
        forward
    )
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(eye, dirs.shape)
    return origins.astype(np.float32), dirs.astype(np.float32)


def intersect_aabb(rays_o, rays_d, bounds):
    """Return near/far distances where rays intersect an axis-aligned box."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    box_min = np.asarray([xmin, ymin, zmin], dtype=np.float32)
    box_max = np.asarray([xmax, ymax, zmax], dtype=np.float32)
    inv_d = 1. / np.where(np.abs(rays_d) < 1e-8, 1e-8, rays_d)
    t0 = (box_min - rays_o) * inv_d
    t1 = (box_max - rays_o) * inv_d
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    near = np.maximum(np.max(tmin, axis=-1), 0.)
    far = np.min(tmax, axis=-1)
    hit = far > near
    return near.astype(np.float32), far.astype(np.float32), hit


def pet_to_rgb(values):
    """Simple PET-like black-red-yellow-white colormap for normalized values."""
    values = np.clip(values, 0., 1.)
    rgb = np.zeros(values.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(values * 3., 0., 1.)
    rgb[..., 1] = np.clip(values * 3. - 1., 0., 1.)
    rgb[..., 2] = np.clip(values * 3. - 2., 0., 1.)
    return rgb


def render_pet_camera(
        H, W, focal, eye, bounds, model, network_query_fn,
        N_samples, chunk, raw_activation='softplus', density_max=0.015):
    rays_o, rays_d = make_camera_rays(H, W, focal, eye)
    rays_o_flat = rays_o.reshape([-1, 3])
    rays_d_flat = rays_d.reshape([-1, 3])
    near, far, hit = intersect_aabb(rays_o_flat, rays_d_flat, bounds)

    image = np.zeros((rays_o_flat.shape[0],), dtype=np.float32)
    acc = np.zeros_like(image)
    hit_ids = np.flatnonzero(hit)
    t_vals = np.linspace(0., 1., N_samples, dtype=np.float32)

    for start in range(0, hit_ids.shape[0], chunk):
        ids = hit_ids[start:start + chunk]
        z_vals = near[ids, None] * (1. - t_vals) + far[ids, None] * t_vals
        points = rays_o_flat[ids, None, :] + rays_d_flat[ids, None, :] * z_vals[..., None]
        raw = network_query_fn(
            tf.convert_to_tensor(points, dtype=tf.float32),
            None,
            model)
        intensity, density = activate_pet_raw(
            raw,
            raw_activation=raw_activation,
            density_max=density_max)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = np.concatenate([dists, dists[:, -1:]], axis=-1)
        alpha = 1. - np.exp(-density.numpy() * dists)
        trans = np.cumprod(
            np.concatenate([np.ones_like(alpha[:, :1]), 1. - alpha + 1e-10],
                           axis=-1),
            axis=-1)[:, :-1]
        weights = alpha * trans
        image[ids] = np.sum(weights * intensity.numpy(), axis=-1)
        acc[ids] = np.sum(weights, axis=-1)

    return image.reshape([H, W]), acc.reshape([H, W])


def render_pet_path(args):
    """Render PET-NeRF from a NeRF-like orbit camera path."""
    import imageio

    if args.ft_weights is None:
        raise ValueError('--ft_weights is required when rendering views.')

    scanner = PETScanner(get_scanner_config(args.scanner))
    enable_fine_model_if_saved(args)
    render_kwargs_train, _, _, models = create_pet_nerf(args)
    model = load_render_model(args.ft_weights, models)

    if args.volume_bounds is None:
        bounds = get_default_volume_bounds(scanner, args.volume_margin)
    else:
        bounds = parse_float6(args.volume_bounds)
    H, W = parse_int2(args.render_hw)
    if args.render_focal is None:
        focal = 0.5 * W / np.tan(0.5 * np.deg2rad(args.render_fov))
    else:
        focal = args.render_focal
    if args.render_radius is None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        radius = 1.6 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    else:
        radius = args.render_radius

    output_dir = args.render_output
    if output_dir is None:
        output_dir = os.path.join(args.basedir, args.expname, 'renderonly_path')
    os.makedirs(output_dir, exist_ok=True)

    print('Rendering PET-NeRF camera path')
    print('  weights:', args.ft_weights)
    print('  output:', output_dir)
    print('  orbit_frames:', args.render_frames)
    print('  topdown_frames:', args.render_topdown_frames)
    print('  hwf:', (H, W, focal))
    print('  bounds:', bounds)

    frames = []
    raw_frames = []
    elevation = np.deg2rad(args.render_elevation)
    camera_eyes = []
    for angle in np.linspace(0., 2. * np.pi, args.render_frames,
                             endpoint=False):
        camera_eyes.append(np.asarray([
            radius * np.cos(elevation) * np.cos(angle),
            radius * np.cos(elevation) * np.sin(angle),
            radius * np.sin(elevation),
        ], dtype=np.float32))
    for _ in range(args.render_topdown_frames):
        camera_eyes.append(np.asarray([0., 0., radius], dtype=np.float32))

    for frame_id, eye in enumerate(camera_eyes):
        image, acc = render_pet_camera(
            H, W, focal, eye, bounds,
            model,
            render_kwargs_train['network_query_fn'],
            args.render_samples,
            args.render_chunk,
            raw_activation=args.raw_activation,
            density_max=args.density_max)
        raw_frames.append(image)
        print('  frame', frame_id, 'range', float(image.min()), float(image.max()),
              'acc', float(acc.min()), float(acc.max()))

    raw_frames = np.asarray(raw_frames, dtype=np.float32)
    scale = np.percentile(raw_frames, args.render_percentile)
    if scale <= 0.:
        scale = float(raw_frames.max()) if raw_frames.max() > 0. else 1.

    for frame_id, image in enumerate(raw_frames):
        rgb = pet_to_rgb(image / scale)
        frame = (255 * np.clip(rgb, 0., 1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, '{:03d}.png'.format(frame_id)), frame)
        frames.append(frame)

    video_path = os.path.join(output_dir, 'video.mp4')
    imageio.mimwrite(video_path, frames, fps=args.render_fps, quality=8)
    print('saved render frames at', output_dir)
    print('saved render video at', video_path)
    print('normalization percentile', args.render_percentile, 'value', float(scale))
    return output_dir


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--expname', type=str, default='pet_nerf_test')
    parser.add_argument('--basedir', type=str, default='./logs')
    parser.add_argument('--datadir', type=str, default=None,
                        help='mich counts by LORID, or npz with crystal1/crystal2/targets')
    parser.add_argument('--scanner', type=str, default='D80',
                        help='scanner name from PETnerf/config.py')
    parser.add_argument('--mich_dtype', type=str, default='float32',
                        help='dtype of each count stored in .mich')
    parser.add_argument('--mich_header_bytes', type=int, default=0,
                        help='bytes to skip before .mich count payload')

    parser.add_argument('--netdepth', type=int, default=8)
    parser.add_argument('--netwidth', type=int, default=256)
    parser.add_argument('--netdepth_fine', type=int, default=8)
    parser.add_argument('--netwidth_fine', type=int, default=256)
    parser.add_argument('--net_skips', type=str, default='4',
                        help='comma-separated hidden layer indices with skip connections')
    parser.add_argument('--N_rand', type=int, default=4096)
    parser.add_argument('--N_iters', type=int, default=1000000)
    parser.add_argument('--lrate', type=float, default=5e-4)
    parser.add_argument('--lrate_decay', type=int, default=250)
    parser.add_argument('--chunk', type=int, default=1024 * 32)
    parser.add_argument('--netchunk', type=int, default=1024 * 64)
    parser.add_argument('--allow_repeated_lor_batch', action='store_true',
                        help='allow repeated LOR ids inside each batch; faster but noisier')
    parser.add_argument('--bin_cut_ratio', type=float, default=0.,
                        help='for .mich LORID sampling, keep only middle bins: '
                             'ratio*bin_num < bin_id < (1-ratio)*bin_num; '
                             '0 disables bin filtering')
    parser.add_argument('--lor_sample_mode', type=str, default='slice',
                        choices=['slice', 'global'],
                        help='for .mich data, sample one slice then rays inside it, '
                             'or sample uniformly over all valid LORs')

    parser.add_argument('--N_samples', type=int, default=0,
                        help='samples per LOR; <=0 means auto from scanner geometry')
    parser.add_argument('--N_importance', type=int, default=0,
                        help='number of additional fine samples per LOR')
    parser.add_argument('--sample_step', type=float, default=0.5,
                        help='maximum spacing in mm between LOR samples when N_samples<=0')
    parser.add_argument('--fov_size', type=str, default=None,
                        help='FOV box size in scanner units, either one value '
                             'for a cube or sx,sy,sz; clips LOR sampling to '
                             'this box during training')
    parser.add_argument('--fov_center', type=str, default='0,0,0',
                        help='center of --fov_size box as x,y,z')
    parser.add_argument('--fov_bounds', type=str, default=None,
                        help='explicit FOV bounds xmin,xmax,ymin,ymax,zmin,zmax; '
                             'overrides --fov_size')
    parser.add_argument('--perturb', type=float, default=1.)
    parser.add_argument('--i_embed', type=int, default=0)
    parser.add_argument('--multires', type=int, default=10)
    parser.add_argument('--multires_angles', type=int, default=4,
                        help='deprecated; ignored because PET fields are direction-independent')
    parser.add_argument('--raw_activation', type=str, default='softplus',
                        choices=['softplus', 'relu'],
                        help='positive activation for activity/intensity output')
    parser.add_argument('--density_max', type=float, default=0.015,
                        help='upper bound for attenuation density/mu; density is sigmoid(raw) * density_max')
    parser.add_argument('--attn_image', type=str, default=None,
                        help='fixed 2D attenuation/mu map; when set, PET-NeRF predicts activity only')
    parser.add_argument('--attn_shape', type=str, default=None,
                        help='attenuation map shape for raw files, height,width or height,width,1')
    parser.add_argument('--attn_dtype', type=str, default='float32',
                        help='dtype for raw attenuation maps')
    parser.add_argument('--attn_header_bytes', type=int, default=0,
                        help='bytes to skip before raw attenuation map payload')
    parser.add_argument('--attn_bounds', type=str, default=None,
                        help='attenuation image bounds xmin,xmax,ymin,ymax; default uses scanner radius')
    parser.add_argument('--attn_activity_soft_mask', action='store_true',
                        help='multiply activity by a soft support mask from fixed mu; disabled by default')
    parser.add_argument('--i_print', type=int, default=100)
    parser.add_argument('--i_weights', type=int, default=20000)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--ft_weights', type=str, default=None,
                        help='model_*.npy weights to reload for volume export')
    parser.add_argument('--export_volume', action='store_true',
                        help='sample a trained PET-NeRF into a 3D npz volume')
    parser.add_argument('--render_views', action='store_true',
                        help='render a NeRF-like orbit video from a trained PET-NeRF')
    parser.add_argument('--volume_output', type=str, default=None,
                        help='output npz path for exported intensity/density volume')
    parser.add_argument('--volume_resolution', type=str, default='128',
                        help='one value or nx,ny,nz voxel counts')
    parser.add_argument('--volume_bounds', type=str, default=None,
                        help='xmin,xmax,ymin,ymax,zmin,zmax in scanner units')
    parser.add_argument('--volume_margin', type=float, default=0.,
                        help='margin added to automatic scanner bounds')
    parser.add_argument('--volume_chunk', type=int, default=1024 * 64,
                        help='number of voxel centers queried per network call')
    parser.add_argument('--volume_direction_mode', type=str, default='six',
                        choices=['z', 'xyz', 'six'],
                        help='deprecated; ignored because exported PET fields are direction-independent')
    parser.add_argument('--render_output', type=str, default=None,
                        help='directory for rendered PNG frames and video.mp4')
    parser.add_argument('--render_frames', type=int, default=40,
                        help='number of orbit frames to render')
    parser.add_argument('--render_topdown_frames', type=int, default=12,
                        help='extra final frames from a top-down camera; 0 disables')
    parser.add_argument('--render_hw', type=str, default='256,256',
                        help='render height,width or one square size')
    parser.add_argument('--render_fov', type=float, default=45.,
                        help='camera horizontal field of view in degrees')
    parser.add_argument('--render_focal', type=float, default=None,
                        help='override focal length in pixels')
    parser.add_argument('--render_radius', type=float, default=None,
                        help='orbit radius; default is derived from volume bounds')
    parser.add_argument('--render_elevation', type=float, default=20.,
                        help='orbit camera elevation in degrees')
    parser.add_argument('--render_samples', type=int, default=192,
                        help='samples per camera ray')
    parser.add_argument('--render_chunk', type=int, default=1024,
                        help='camera rays rendered per network batch')
    parser.add_argument('--render_fps', type=int, default=30,
                        help='output video frame rate')
    parser.add_argument('--render_percentile', type=float, default=99.5,
                        help='percentile used to normalize rendered intensity')

    return parser


def train(argv=None):
    parser = config_parser()
    args = parser.parse_args(argv)

    scanner = PETScanner(get_scanner_config(args.scanner))
    args.fov_bounds_parsed = get_fov_bounds(args)
    args.net_skips_parsed = parse_int_list(args.net_skips)
    args.attn_shape_parsed = (
        parse_int2_or_3(args.attn_shape)
        if args.attn_shape is not None else None)
    args.attn_bounds_parsed = get_attn_bounds(args, scanner)

    if args.render_views:
        return render_pet_path(args)

    if args.export_volume:
        return export_pet_volume(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    if args.N_samples <= 0:
        if args.fov_bounds_parsed is None:
            args.N_samples, max_lor_length = estimate_n_samples_from_scanner(
                scanner, args.sample_step)
            sample_source = 'scanner geometry'
        else:
            args.N_samples, max_lor_length = estimate_n_samples_from_bounds(
                args.fov_bounds_parsed, args.sample_step)
            sample_source = 'FOV bounds'
        print('Auto N_samples', args.N_samples,
              'from', sample_source,
              'for max_lor_length {:.3f} mm and sample_step {:.3f} mm'.format(
                  max_lor_length, args.sample_step))
    if args.fov_bounds_parsed is not None:
        print('Training FOV bounds', args.fov_bounds_parsed)

    use_lor_ids = False
    if args.datadir is None:
        print('No PET data passed; using demo synthetic LORs.')
        lors, targets = make_demo_lors()
    elif args.datadir.endswith('.npz'):
        lors, targets = load_pet_npz(args.datadir)
    else:
        targets = load_mich_counts(
            args.datadir, scanner,
            dtype=args.mich_dtype,
            header_bytes=args.mich_header_bytes)
        use_lor_ids = True
        lors = None

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    render_kwargs_train, _, grad_vars, models = create_pet_nerf(args)

    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(
            lrate, decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    N = scanner.get_lor_num() if use_lor_ids else lors.shape[0]
    bin_sample_kwargs = {}
    if use_lor_ids:
        bin_num = scanner.get_bin_num()
        view_num = scanner.get_view_num()
        slice_num = scanner.get_slice_num()
        bin_start, bin_stop = get_bin_cut_range(bin_num, args.bin_cut_ratio)
        valid_lor_num = slice_num * view_num * (bin_stop - bin_start)
        bin_sample_kwargs = {
            'bin_num': bin_num,
            'view_num': view_num,
            'bin_start': bin_start,
            'bin_stop': bin_stop,
            'sample_mode': args.lor_sample_mode,
        }
        print('LOR sampling',
              'mode', args.lor_sample_mode,
              'dims(bin,view,slice)', (bin_num, view_num, slice_num),
              'ratio', args.bin_cut_ratio,
              'bin_range', (bin_start, bin_stop),
              'valid_bins', bin_stop - bin_start,
              'valid_LORs', valid_lor_num,
              'of', scanner.get_lor_num())
    print('Scanner', args.scanner,
          'crystals', scanner.get_crystal_num(),
          'LORs', scanner.get_lor_num())
    print('Loaded PET targets', targets.shape, 'use_lor_ids', use_lor_ids)
    print('Begin PET-NeRF training')

    def save_models(step):
        for name, model in models.items():
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(name, step))
            np.save(path, np.array(model.get_weights(), dtype=object))
            print('saved weights at', path)

    for i in range(args.N_iters):
        time0 = time.time()
        select_inds = sample_lor_indices(
            N,
            args.N_rand,
            allow_repeated=args.allow_repeated_lor_batch,
            **bin_sample_kwargs)
        if use_lor_ids:
            crystal1, crystal2 = scanner.get_lor_endpoint_array(select_inds)
            batch_lors_np = np.concatenate([crystal1, crystal2], axis=-1)
        else:
            batch_lors_np = lors[select_inds]
        batch_lors = tf.convert_to_tensor(batch_lors_np.astype(np.float32))
        target_s = tf.convert_to_tensor(
            np.asarray(targets[select_inds], dtype=np.float32).reshape([-1, 1]))

        with tf.GradientTape() as tape:
            outputs = render_lors(batch_lors, **render_kwargs_train)
            loss = poisson_nll(outputs['count'], target_s)
            if 'count0' in outputs:
                loss += poisson_nll(outputs['count0'], target_s)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        if i % args.i_weights == 0:
            save_models(i)

        if i % args.i_print == 0 or i < 10:
            dt = time.time() - time0
            print(
                expname, i,
                'poisson_nll', loss.numpy(),
                'target_mean', float(tf.reduce_mean(target_s)),
                'count_range',
                float(tf.reduce_min(outputs['count'])),
                float(tf.reduce_max(outputs['count'])),
                'intensity_alpha_sum_range',
                float(tf.reduce_min(outputs['intensity_alpha_sum'])),
                float(tf.reduce_max(outputs['intensity_alpha_sum'])),
                'density_integral_range',
                float(tf.reduce_min(outputs['density_integral'])),
                float(tf.reduce_max(outputs['density_integral'])),
                'time', dt)

    if args.N_iters > 0 and args.N_iters % args.i_weights == 0:
        save_models(args.N_iters)


if __name__ == '__main__':
    train()
