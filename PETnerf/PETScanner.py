from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


@dataclass
class Volume3D:
    x: float = 0
    y: float = 0
    z: float = 0

    @classmethod
    def from_values(cls, values):
        if isinstance(values, cls):
            return values
        return cls(values[0], values[1], values[2])

    def as_array(self, dtype=np.float32):
        return np.array([self.x, self.y, self.z], dtype=dtype)


class PETScanner:
    """Config-driven Python port of UPETScanner geometry/index utilities.

    This class intentionally does not include the C++ InitD80Scanner,
    InitD180Scanner, etc. presets. Build an instance from config instead.
    """

    def __init__(self, config=None):
        self.crystal_num = Volume3D()
        self.block_num = Volume3D()
        self.module_num = Volume3D()

        self.crystal_size = Volume3D()
        self.block_size = Volume3D()
        self.module_size = Volume3D()
        self.panel_size = Volume3D()

        self.crystal_pitch = Volume3D()
        self.block_pitch = Volume3D()
        self.module_pitch = Volume3D()
        self.panel_pitch = Volume3D()

        self.panel_num = 0
        self.crystal_clockwise_offset = 0
        self.scanner_radius = 0.0
        self._crystal_centers_cache = None

        if config is not None:
            self.configure(config)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def configure(self, config):
        self._crystal_centers_cache = None
        self.set_crystal_num(*config["crystal_num"])
        self.set_block_num(*config["block_num"])
        self.set_module_num(*config["module_num"])
        self.set_crystal_size(*config["crystal_size"])
        self.set_block_size(*config["block_size"])
        self.set_module_size(*config["module_size"])
        self.set_panel_size(*config["panel_size"])
        self.set_crystal_pitch(*config["crystal_pitch"])
        self.set_block_pitch(*config["block_pitch"])
        self.set_module_pitch(*config["module_pitch"])

        panel_pitch = config.get("panel_pitch")
        if panel_pitch is None:
            self.panel_pitch = Volume3D.from_values(self.panel_size)
        else:
            self.panel_pitch = Volume3D.from_values(panel_pitch)

        self.set_ring_para(
            config["panel_num"],
            config.get("crystal_clockwise_offset", 0),
            config["scanner_radius"],
        )
        return self

    def set_crystal_num(self, num_x, num_y, num_z):
        self.crystal_num = Volume3D(int(num_x), int(num_y), int(num_z))

    def set_block_num(self, num_x, num_y, num_z):
        self.block_num = Volume3D(int(num_x), int(num_y), int(num_z))

    def set_module_num(self, num_x, num_y, num_z):
        self.module_num = Volume3D(int(num_x), int(num_y), int(num_z))

    def set_crystal_size(self, size_x, size_y, size_z):
        self.crystal_size = Volume3D(float(size_x), float(size_y), float(size_z))

    def set_block_size(self, size_x, size_y, size_z):
        self.block_size = Volume3D(float(size_x), float(size_y), float(size_z))

    def set_module_size(self, size_x, size_y, size_z):
        self.module_size = Volume3D(float(size_x), float(size_y), float(size_z))

    def set_panel_size(self, size_x, size_y, size_z):
        self.panel_size = Volume3D(float(size_x), float(size_y), float(size_z))

    def set_crystal_pitch(self, pitch_x, pitch_y, pitch_z):
        self.crystal_pitch = Volume3D(float(pitch_x), float(pitch_y), float(pitch_z))

    def set_block_pitch(self, pitch_x, pitch_y, pitch_z):
        self.block_pitch = Volume3D(float(pitch_x), float(pitch_y), float(pitch_z))

    def set_module_pitch(self, pitch_x, pitch_y, pitch_z):
        self.module_pitch = Volume3D(float(pitch_x), float(pitch_y), float(pitch_z))

    def set_ring_para(self, panel_num, crystal_clockwise_offset, scanner_radius):
        self.panel_num = int(panel_num)
        self.crystal_clockwise_offset = int(crystal_clockwise_offset)
        self.scanner_radius = float(scanner_radius)

    def get_crystal_num_one_ring(self):
        return int(self.panel_num * self.module_num.y * self.block_num.y *
                   self.crystal_num.y)

    def get_ring_num(self):
        return int(self.module_num.z * self.block_num.z * self.crystal_num.z)

    def get_block_num_one_ring(self):
        return int(self.panel_num * self.module_num.y * self.block_num.y)

    def get_block_ring_num(self):
        return int(self.module_num.z * self.block_num.z)

    def get_bin_num(self):
        return self.get_crystal_num_one_ring() - 1

    def get_view_num(self):
        return self.get_crystal_num_one_ring() // 2

    def get_slice_num(self):
        return self.get_ring_num() * self.get_ring_num()

    def get_lor_num(self):
        return self.get_bin_num() * self.get_view_num() * self.get_slice_num()

    def get_crystal_num(self):
        return self.get_crystal_num_one_ring() * self.get_ring_num()

    def get_block_num(self):
        return self.get_block_num_one_ring() * self.get_block_ring_num()

    def get_crystal_num_z_in_panel(self):
        return self.get_ring_num()

    def get_crystal_num_y_in_panel(self):
        return int(self.crystal_num.y * self.block_num.y * self.module_num.y)

    def get_crystal_num_z_in_module(self):
        return int(self.crystal_num.z * self.block_num.z)

    def get_crystal_num_y_in_module(self):
        return int(self.crystal_num.y * self.block_num.y)

    def get_crystal_num_z_in_block(self):
        return int(self.crystal_num.z)

    def get_crystal_num_y_in_block(self):
        return int(self.crystal_num.y)

    def get_block_num_z_in_panel(self):
        return int(self.block_num.z * self.module_num.z)

    def get_block_num_y_in_panel(self):
        return int(self.block_num.y * self.module_num.y)

    def get_block_num_z_in_module(self):
        return int(self.block_num.z)

    def get_block_num_y_in_module(self):
        return int(self.block_num.y)

    def get_module_num_z_in_panel(self):
        return int(self.module_num.z)

    def get_module_num_y_in_panel(self):
        return int(self.module_num.y)

    def get_panel_num(self):
        return self.panel_num

    def get_length_z(self):
        return self.module_pitch.z * self.module_num.z

    def get_panel_size_z(self):
        return self.panel_size.z

    def get_panel_size_y(self):
        return self.panel_size.y

    def get_module_pitch_z(self):
        return self.module_pitch.z

    def get_module_pitch_y(self):
        return self.module_pitch.y

    def get_module_size_z(self):
        return self.module_size.z

    def get_module_size_y(self):
        return self.module_size.y

    def get_block_pitch_z(self):
        return self.block_pitch.z

    def get_block_pitch_y(self):
        return self.block_pitch.y

    def get_block_size_z(self):
        return self.block_size.z

    def get_block_size_y(self):
        return self.block_size.y

    def get_crystal_pitch_y(self):
        return self.crystal_pitch.y

    def get_crystal_pitch_z(self):
        return self.crystal_pitch.z

    def get_crystal_size_y(self):
        return self.crystal_size.y

    def get_crystal_size_z(self):
        return self.crystal_size.z

    def get_radius(self):
        return self.scanner_radius

    def get_block_in_ring_from_crystal_id(self, crystal_id):
        cry = crystal_id % self.get_crystal_num_one_ring()
        return cry // self.get_crystal_num_y_in_block()

    def get_crystal_clockwise_offset(self):
        return self.crystal_clockwise_offset

    def get_ring_min_coordinate_z(self, ring):
        ring = int(ring)
        z = (
            (ring // self.get_crystal_num_z_in_module()) * self.module_pitch.z
            + (self.module_pitch.z - self.module_size.z) / 2
            + ((ring % self.get_crystal_num_z_in_module())
               // self.get_crystal_num_z_in_block()) * self.block_pitch.z
            + (self.block_pitch.z - self.block_size.z) / 2
            + (ring % self.get_crystal_num_z_in_block()) * self.crystal_pitch.z
            + (self.crystal_pitch.z - self.crystal_size.z) / 2
            - self.get_length_z() / 2
        )
        return z

    def get_crystal_position(self, crystal_in_ring, ring):
        """Return the four y-z face vertices of one crystal as [4, 3]."""

        crystal_in_ring = (
            int(crystal_in_ring) - self.crystal_clockwise_offset
            + self.get_crystal_num_one_ring()
        ) % self.get_crystal_num_one_ring()
        ring = int(ring)

        z0 = self.get_ring_min_coordinate_z(ring)
        z1 = z0 + self.crystal_size.z

        panel = crystal_in_ring // self.get_crystal_num_y_in_panel()
        module = ((crystal_in_ring % self.get_crystal_num_y_in_panel())
                  // self.get_crystal_num_y_in_module())
        block = ((crystal_in_ring % self.get_crystal_num_y_in_module())
                 // self.get_crystal_num_y_in_block())
        cry = crystal_in_ring % self.get_crystal_num_y_in_block()

        angle = panel * 2 * pi / self.panel_num
        panel_offset = (
            module * self.module_pitch.y
            + (self.module_pitch.y - self.module_size.y) / 2
            + block * self.block_pitch.y
            + (self.block_pitch.y - self.block_size.y) / 2
            + cry * self.crystal_pitch.y
            + (self.crystal_pitch.y - self.crystal_size.y) / 2
            - self.get_panel_size_y() / 2
        )

        p0 = np.array([
            self.scanner_radius * cos(angle)
            + panel_offset * cos(angle + pi / 2),
            self.scanner_radius * sin(angle)
            + panel_offset * sin(angle + pi / 2),
            z0,
        ], dtype=np.float32)
        p1 = np.array([
            self.scanner_radius * cos(angle)
            + (panel_offset + self.crystal_size.y) * cos(angle + pi / 2),
            self.scanner_radius * sin(angle)
            + (panel_offset + self.crystal_size.y) * sin(angle + pi / 2),
            z0,
        ], dtype=np.float32)
        p2 = p1.copy()
        p2[2] = z1
        p3 = p0.copy()
        p3[2] = z1
        return np.stack([p0, p1, p2, p3], axis=0)

    def get_crystal_center(self, crystal_in_ring, ring):
        return self.get_crystal_position(crystal_in_ring, ring).mean(axis=0)

    def get_crystal_center_from_global_id(self, crystal_id):
        crystal_one_ring = self.get_crystal_num_one_ring()
        crystal_in_ring = int(crystal_id) % crystal_one_ring
        ring = int(crystal_id) // crystal_one_ring
        return self.get_crystal_center(crystal_in_ring, ring)

    def get_crystal_id_from_lor_id(self, lor_id):
        crystal_one_ring = self.get_crystal_num_one_ring()
        bin_num = self.get_bin_num()
        view_num = self.get_view_num()

        lor_id = int(lor_id)
        bin_id = lor_id % bin_num
        view = (lor_id // bin_num) % view_num
        slice_id = lor_id // (bin_num * view_num)

        cry1, cry2 = self.get_crystal_id_in_ring_from_view_bin(view, bin_id)
        ring1, ring2 = self.get_ring1_ring2_from_slice(slice_id)
        crystal_id1 = cry1 + ring1 * crystal_one_ring
        crystal_id2 = cry2 + ring2 * crystal_one_ring
        return crystal_id1, crystal_id2

    def get_crystal_id_array_from_lor_id(self, lor_ids):
        """Vectorized LORID -> global crystalID1/crystalID2."""
        lor_ids = np.asarray(lor_ids, dtype=np.int64)
        crystal_one_ring = self.get_crystal_num_one_ring()
        bin_num = self.get_bin_num()
        view_num = self.get_view_num()
        ring_num = self.get_ring_num()

        bin_id = lor_ids % bin_num
        view = (lor_ids // bin_num) % view_num
        slice_id = lor_ids // (bin_num * view_num)

        cry2 = bin_id // 2 + 1
        cry1 = crystal_one_ring + (1 - bin_id % 2) - cry2
        cry2 = (cry2 + view) % crystal_one_ring
        cry1 = (cry1 + view) % crystal_one_ring

        ring1 = slice_id // ring_num
        ring2 = slice_id % ring_num
        crystal_id1 = cry1 + ring1 * crystal_one_ring
        crystal_id2 = cry2 + ring2 * crystal_one_ring
        return crystal_id1.astype(np.int64), crystal_id2.astype(np.int64)

    def get_lor_id_from_crystal_id(self, crystal_id1, crystal_id2):
        crystal_one_ring = self.get_crystal_num_one_ring()
        cry1 = int(crystal_id1) % crystal_one_ring
        cry2 = int(crystal_id2) % crystal_one_ring
        ring1 = int(crystal_id1) // crystal_one_ring
        ring2 = int(crystal_id2) // crystal_one_ring
        return self.get_lor_id_from_ring_and_crystal_in_ring(
            ring1, cry1, ring2, cry2)

    def get_ring1_ring2_from_slice(self, slice_id):
        slice_id = int(slice_id)
        ring1 = slice_id // self.get_ring_num()
        ring2 = slice_id % self.get_ring_num()
        return ring1, ring2

    def get_crystal_id_in_ring_from_view_bin(self, view, bin_id):
        crystal_one_ring = self.get_crystal_num_one_ring()
        bin_id = int(bin_id)
        view = int(view)

        cry2 = bin_id // 2 + 1
        cry1 = crystal_one_ring + (1 - bin_id % 2) - cry2
        cry2 = (cry2 + view) % crystal_one_ring
        cry1 = (cry1 + view) % crystal_one_ring
        return cry1, cry2

    def get_lor_id_from_ring_and_crystal_in_ring(self, ring1, cry1, ring2, cry2):
        crystal_one_ring = self.get_crystal_num_one_ring()
        ring1 = int(ring1)
        ring2 = int(ring2)
        cry1 = int(cry1)
        cry2 = int(cry2)

        view = ((cry1 + cry2) % crystal_one_ring) // 2

        cry1_shifted = cry1 - view
        cry2_shifted = cry2 - view
        if cry1_shifted <= 0:
            cry1_shifted += crystal_one_ring
        if cry2_shifted <= 0:
            cry2_shifted += crystal_one_ring

        if cry1_shifted > cry2_shifted:
            cry1_real = cry1_shifted
            cry2_real = cry2_shifted
            ring1_real = ring1
            ring2_real = ring2
        else:
            cry1_real = cry2_shifted
            cry2_real = cry1_shifted
            ring1_real = ring2
            ring2_real = ring1

        bin_id = (crystal_one_ring - 1) - (cry1_real - cry2_real)
        slice_id = ring1_real * self.get_ring_num() + ring2_real
        return (slice_id * (crystal_one_ring - 1) * (crystal_one_ring // 2)
                + view * (crystal_one_ring - 1) + bin_id)

    def is_good_pair(self, panel1, panel2, min_sector_difference):
        panel_diff = abs(int(panel1) - int(panel2))
        return int(panel_diff >= min_sector_difference and
                   self.panel_num - panel_diff >= min_sector_difference)

    def get_lor_endpoints(self, lor_id):
        crystal_id1, crystal_id2 = self.get_crystal_id_from_lor_id(lor_id)
        crystal1 = self.get_crystal_center_from_global_id(crystal_id1)
        crystal2 = self.get_crystal_center_from_global_id(crystal_id2)
        return crystal1, crystal2

    def get_lor_endpoint_array(self, lor_ids):
        crystal_id1, crystal_id2 = self.get_crystal_id_array_from_lor_id(lor_ids)
        centers = self.get_all_crystal_centers()
        crystal1 = centers[crystal_id1]
        crystal2 = centers[crystal_id2]
        return crystal1, crystal2

    def get_all_crystal_centers(self):
        if self._crystal_centers_cache is not None:
            return self._crystal_centers_cache
        centers = [
            self.get_crystal_center_from_global_id(crystal_id)
            for crystal_id in range(self.get_crystal_num())
        ]
        self._crystal_centers_cache = np.stack(centers, axis=0)
        return self._crystal_centers_cache


# C++-style aliases for easier cross-checking against UPETScanner.cpp.
PETScanner.SetCrystalNum = PETScanner.set_crystal_num
PETScanner.SetBlockNum = PETScanner.set_block_num
PETScanner.SetModuleNum = PETScanner.set_module_num
PETScanner.SetCrystalSize = PETScanner.set_crystal_size
PETScanner.SetBlockSize = PETScanner.set_block_size
PETScanner.SetModuleSize = PETScanner.set_module_size
PETScanner.SetPanelSize = PETScanner.set_panel_size
PETScanner.SetCrystalPitch = PETScanner.set_crystal_pitch
PETScanner.SetBlockPitch = PETScanner.set_block_pitch
PETScanner.SetModulePitch = PETScanner.set_module_pitch
PETScanner.SetRingPara = PETScanner.set_ring_para

PETScanner.GetCrystalNumOneRing = PETScanner.get_crystal_num_one_ring
PETScanner.GetRingNum = PETScanner.get_ring_num
PETScanner.GetBlockNumOneRing = PETScanner.get_block_num_one_ring
PETScanner.GetBlockRingNum = PETScanner.get_block_ring_num
PETScanner.GetBinNum = PETScanner.get_bin_num
PETScanner.GetViewNum = PETScanner.get_view_num
PETScanner.GetSliceNum = PETScanner.get_slice_num
PETScanner.GetLORNum = PETScanner.get_lor_num
PETScanner.GetCrystalNum = PETScanner.get_crystal_num
PETScanner.GetBlockNum = PETScanner.get_block_num
PETScanner.GetCrystalNumZInPanel = PETScanner.get_crystal_num_z_in_panel
PETScanner.GetCrystalNumYInPanel = PETScanner.get_crystal_num_y_in_panel
PETScanner.GetCrystalNumZInModule = PETScanner.get_crystal_num_z_in_module
PETScanner.GetCrystalNumYInModule = PETScanner.get_crystal_num_y_in_module
PETScanner.GetCrystalNumZInBlock = PETScanner.get_crystal_num_z_in_block
PETScanner.GetCrystalNumYInBlock = PETScanner.get_crystal_num_y_in_block
PETScanner.GetBlockNumZInPanel = PETScanner.get_block_num_z_in_panel
PETScanner.GetBlockNumYInPanel = PETScanner.get_block_num_y_in_panel
PETScanner.GetBlockNumZInModule = PETScanner.get_block_num_z_in_module
PETScanner.GetBlockNumYInModule = PETScanner.get_block_num_y_in_module
PETScanner.GetModuleNumZInPanel = PETScanner.get_module_num_z_in_panel
PETScanner.GetModuleNumYInPanel = PETScanner.get_module_num_y_in_panel
PETScanner.GetPanelNum = PETScanner.get_panel_num
PETScanner.GetLengthZ = PETScanner.get_length_z
PETScanner.GetPanelSizeZ = PETScanner.get_panel_size_z
PETScanner.GetPanelSizeY = PETScanner.get_panel_size_y
PETScanner.GetModulePitchZ = PETScanner.get_module_pitch_z
PETScanner.GetModulePitchY = PETScanner.get_module_pitch_y
PETScanner.GetModuleSizeZ = PETScanner.get_module_size_z
PETScanner.GetModuleSizeY = PETScanner.get_module_size_y
PETScanner.GetBlockPitchZ = PETScanner.get_block_pitch_z
PETScanner.GetBlockPitchY = PETScanner.get_block_pitch_y
PETScanner.GetBlockSizeZ = PETScanner.get_block_size_z
PETScanner.GetBlockSizeY = PETScanner.get_block_size_y
PETScanner.GetCrystalPitchY = PETScanner.get_crystal_pitch_y
PETScanner.GetCrystalPitchZ = PETScanner.get_crystal_pitch_z
PETScanner.GetCrystalSizeY = PETScanner.get_crystal_size_y
PETScanner.GetCrystalSizeZ = PETScanner.get_crystal_size_z
PETScanner.GetRadius = PETScanner.get_radius
PETScanner.GetBlockInRingFromCrystalId = PETScanner.get_block_in_ring_from_crystal_id
PETScanner.GetCrystalClockwiseOffset = PETScanner.get_crystal_clockwise_offset
PETScanner.GetCrystalPosition = PETScanner.get_crystal_position
PETScanner.GetRingMinCoordinateZ = PETScanner.get_ring_min_coordinate_z
PETScanner.GetCrystalIDFromLORID = PETScanner.get_crystal_id_from_lor_id
PETScanner.GetLORIDFromCrystalID = PETScanner.get_lor_id_from_crystal_id
PETScanner.GetRing1Ring2FromSlice = PETScanner.get_ring1_ring2_from_slice
PETScanner.GetCrystalIDInRingFromViewBin = PETScanner.get_crystal_id_in_ring_from_view_bin
PETScanner.GetLORIDFromRingAndCrystalInRing = (
    PETScanner.get_lor_id_from_ring_and_crystal_in_ring)
PETScanner.IsGoodPair = PETScanner.is_good_pair
