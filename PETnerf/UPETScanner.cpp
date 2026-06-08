/**
 * @file      UPETScanner.cpp
 * @brief     The source file for the UPETScanner class.
 * @author    Ang Li
 * @date      2019-6-8
 * @copyright (C) RAYSOLUTION Co., Ltd. All rights reserved. This software and its associated documentation files are the exclusive property of RAYSOLUTION Co., Ltd. The original development of this software was carried out by PETLab. PETLab retains the right to perform subsequent development on this software for RAYSOLUTION Co., Ltd. under the terms and conditions of the agreement between the two parties.
 * @attention
 * Edit History:
 * 2019-6-8; Ang Li; Create.
 */
#include "UPETScanner.h"
#include <stdlib.h>
#include <math.h>
#include "UVolume.h"

#define PI (3.141592654)

void UPETScanner::SetCrystalNum(int numX, int numY, int numZ)
{
    crystalNum.set(numX, numY, numZ);
}

void UPETScanner::SetBlockNum(int numX, int numY, int numZ)
{
    blockNum.set(numX, numY, numZ);
}

void UPETScanner::SetModuleNum(int numX, int numY, int numZ)
{
    moduleNum.set(numX, numY, numZ);
}

void UPETScanner::SetCrystalSize(double sizeX, double sizeY, double sizeZ)
{
    crystalSize.set(sizeX, sizeY, sizeZ);
}

void UPETScanner::SetBlockSize(double sizeX, double sizeY, double sizeZ)
{
    blockSize.set(sizeX, sizeY, sizeZ);
}

void UPETScanner::SetModuleSize(double sizeX, double sizeY, double sizeZ)
{
    moduleSize.set(sizeX, sizeY, sizeZ);
}

void UPETScanner::SetPanelSize(double sizeX, double sizeY, double sizeZ)
{
    panelSize.set(sizeX, sizeY, sizeZ);
}

void UPETScanner::SetCrystalPitch(double pitchX, double pitchY, double pitchZ)
{
    crystalPitch.set(pitchX, pitchY, pitchZ);
}

void UPETScanner::SetBlockPitch(double pitchX, double pitchY, double pitchZ)
{
    blockPitch.set(pitchX, pitchY, pitchZ);
}

void UPETScanner::SetModulePitch(double pitchX, double pitchY, double pitchZ)
{
    modulePitch.set(pitchX, pitchY, pitchZ);
}

void UPETScanner::SetRingPara(int nPanelNum, int nCrystalClockwiseOffset, double fScannerRadius)
{
    panelNum = nPanelNum;
    crystalClockwiseOffset = nCrystalClockwiseOffset;
    scannerRadius = fScannerRadius;
}

void UPETScanner::InitD80Scanner()
{
    crystalNum.set(1, 13, 13);
    crystalSize.set(13, 1.89, 1.89);
    crystalPitch.set(13, 2.0, 2.0);

    blockNum.set(1, 1, 4);
    blockSize.set(13.3, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(13.3, 26.5, 26.5);

    moduleNum.set(1, 1, 1);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch.set(20, moduleSize.y, moduleSize.z + 2);

    panelNum = 12;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 1;
    scannerRadius = 105.8 / 2;
}

void UPETScanner::InitD180Scanner()
{
    crystalNum.set(1, 13, 13);
    crystalSize.set(13, 1.89, 1.89);
    crystalPitch.set(13, 2.0, 2.0);

    blockNum.set(1, 1, 4);
    blockSize.set(13.3, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(13.3, 26.5, 26.5);

    moduleNum.set(1, 1, 1);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch.set(20, moduleSize.y, moduleSize.z + 2);

    panelNum = 24;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 1;
    scannerRadius = 106.5; // real is 106.5
}

void UPETScanner::InitE180Scanner()
{
    crystalNum.set(1, 13, 13);
    crystalSize.set(13, 1.89, 1.89);
    crystalPitch.set(13, 2.0, 2.0);

    blockNum.set(1, 1, 4);
    blockSize.set(13.3, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(13.3, 26.5, 26.5);

    moduleNum.set(1, 1, 2);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch.set(20, moduleSize.y, moduleSize.z + 2.0);

    panelNum = 24;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 1;
    scannerRadius = 106.5; // real is 106.5
}

void UPETScanner::InitBrainScanner()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 1, 4);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5); // 20200923Updated: to simulate real scanner

    moduleNum.set(1, 1, 2);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 44;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 188.5;
}

void UPETScanner::InitDPET100Scanner()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 1, 4);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5); // 20200923Updated: to simulate real scanner

    moduleNum.set(1, 1, 1);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 88;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 378.5; // 20201110Updated: from 378.0 to 378.5
}

void UPETScanner::InitDPET200Scanner()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 1, 4);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5); // 20200923Updated: to simulate real scanner

    moduleNum.set(1, 1, 2);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 88;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 378.5; // 20201110Updated: from 378.0 to 378.5
}

void UPETScanner::InitDigitMI930()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 2, 4);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5);

    moduleNum.set(1, 1, 3);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 48;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 807.6 / 2;
}

void UPETScanner::InitDigitMI925()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 2, 2);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5);

    moduleNum.set(1, 1, 5);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 48;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 807.6 / 2;
}

void UPETScanner::InitDigitMI920()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 2, 4);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5);

    moduleNum.set(1, 1, 2);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 48;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 807.6 / 2;
}

void UPETScanner::InitDigitMI930_24()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 2, 4);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5);

    moduleNum.set(1, 2, 3);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 24;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 821.8 / 2;
}

void UPETScanner::InitDigitMIi30()
{
    crystalNum.set(1, 6, 6);
    crystalSize.set(20, 3.9, 3.9);
    crystalPitch.set(20, 4.2, 4.2);

    blockNum.set(1, 2, 2);
    blockSize.set(20, crystalPitch.y * crystalNum.y, crystalPitch.z * crystalNum.z);
    blockPitch.set(20, 25.5, 25.5);

    moduleNum.set(1, 1, 5);
    moduleSize.set(20, blockPitch.y * blockNum.y, blockPitch.z * blockNum.z);
    modulePitch = moduleSize;

    panelNum = 24;
    panelSize.set(20, modulePitch.y * moduleNum.y, modulePitch.z * moduleNum.z);
    panelPitch = panelSize;

    crystalClockwiseOffset = 0;
    scannerRadius = 403.7 / 2;
}

int UPETScanner::GetCrystalNumOneRing() const
{
    return panelNum * moduleNum.y * blockNum.y * crystalNum.y;
}

int UPETScanner::GetRingNum() const
{
    return moduleNum.z * blockNum.z * crystalNum.z;
}

int UPETScanner::GetBlockNumOneRing() const
{
    return panelNum * moduleNum.y * blockNum.y;
}

int UPETScanner::GetBlockRingNum() const
{
    return moduleNum.z * blockNum.z;
}

int UPETScanner::GetBinNum() const
{
    return GetCrystalNumOneRing() - 1;
}

int UPETScanner::GetViewNum() const
{
    return GetCrystalNumOneRing() / 2;
}

int UPETScanner::GetSliceNum() const
{
    return GetRingNum() * GetRingNum();
}

size_t UPETScanner::GetLORNum() const
{
    return size_t(GetBinNum()) * size_t(GetViewNum()) * size_t(GetSliceNum());
}

int UPETScanner::GetCrystalNum() const
{
    return GetCrystalNumOneRing() * GetRingNum();
}

int UPETScanner::GetBlockNum() const
{
    return GetBlockNumOneRing() * GetBlockRingNum();
}

int UPETScanner::GetCrystalNumZInPanel() const
{
    return GetRingNum();
}

int UPETScanner::GetCrystalNumYInPanel() const
{
    return crystalNum.y * blockNum.y * moduleNum.y;
}

int UPETScanner::GetCrystalNumZInModule() const
{
    return crystalNum.z * blockNum.z;
}

int UPETScanner::GetCrystalNumYInModule() const
{
    return crystalNum.y * blockNum.y;
}

int UPETScanner::GetCrystalNumZInBlock() const
{
    return crystalNum.z;
}

int UPETScanner::GetCrystalNumYInBlock() const
{
    return crystalNum.y;
}

int UPETScanner::GetBlockNumZInPanel() const // 20200720Updated
{
    return blockNum.z * moduleNum.z;
}
int UPETScanner::GetBlockNumYInPanel() const
{
    return blockNum.y * moduleNum.y;
}
int UPETScanner::GetBlockNumZInModule() const
{
    return blockNum.z;
}
int UPETScanner::GetBlockNumYInModule() const
{
    return blockNum.y;
}
int UPETScanner::GetModuleNumZInPanel() const
{
    return moduleNum.z;
}
int UPETScanner::GetModuleNumYInPanel() const
{
    return moduleNum.y;
}

int UPETScanner::GetPanelNum() const
{
    return panelNum;
}

double UPETScanner::GetLengthZ() const
{
    return modulePitch.z * moduleNum.z;
}

double UPETScanner::GetPanelSizeZ() const
{
    return panelSize.z;
}

double UPETScanner::GetPanelSizeY() const
{
    return panelSize.y;
}

double UPETScanner::GetModulePitchZ() const
{
    return modulePitch.z;
}

double UPETScanner::GetModulePitchY() const
{
    return modulePitch.y;
}

double UPETScanner::GetModuleSizeZ() const
{
    return moduleSize.z;
}

double UPETScanner::GetModuleSizeY() const
{
    return moduleSize.y;
}

double UPETScanner::GetBlockPitchZ() const
{
    return blockPitch.z;
}

double UPETScanner::GetBlockPitchY() const
{
    return blockPitch.y;
}

double UPETScanner::GetBlockSizeZ() const
{
    return blockSize.z;
}

double UPETScanner::GetBlockSizeY() const
{
    return blockSize.y;
}

double UPETScanner::GetCrystalPitchY() const
{
    return crystalPitch.y;
}

double UPETScanner::GetCrystalPitchZ() const
{
    return crystalPitch.z;
}

double UPETScanner::GetCrystalSizeY() const
{
    return crystalSize.y;
}

double UPETScanner::GetCrystalSizeZ() const
{
    return crystalSize.z;
}

double UPETScanner::GetRadius() const
{
    return scannerRadius;
}

int UPETScanner::GetBlockInRingFromCrystalId(int crystalId) const
{
    int cry = crystalId % GetCrystalNumOneRing();
    return cry / GetCrystalNumYInBlock();
}

int UPETScanner::GetCrystalClockwiseOffset() const
{
    return crystalClockwiseOffset;
}

void UPETScanner::GetCrystalPosition(int crystalInRing, int ring, UVolume3D<double> *crystalPos) const
{
    // 异常处理
    // if (ring < 0 || ring > GetRingNum() - 1 || crystalInRing < 0 || crystalInRing > GetCrystalNumOneRing() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetCrystalPosition. Input parameters out of range.\n");
    // 	// 输入参数超出范围，给晶体顶点坐标赋值
    // 	crystalPos[0].set(0, 0, 0);
    // 	crystalPos[1].set(0, 0, 0);
    // 	crystalPos[2].set(0, 0, 0);
    // 	crystalPos[3].set(0, 0, 0);
    // 	return false;
    // }
    // 对晶体在环上的索引进行修正，考虑晶体排列方式
    crystalInRing = (crystalInRing - crystalClockwiseOffset + GetCrystalNumOneRing()) % GetCrystalNumOneRing();
    // 计算环的最小 z 坐标
    double z = GetRingMinCoordinateZ(ring);

    // 设置晶体的四个顶点的 z 坐标
    crystalPos[0].z = z;
    crystalPos[1].z = crystalPos[0].z;
    crystalPos[2].z = z + crystalSize.z;
    crystalPos[3].z = crystalPos[2].z;

    // 计算晶体在 panel、module、block、cry 级别的索引
    int panel1, module1, block1, cry1;
    panel1 = crystalInRing / GetCrystalNumYInPanel();                             // 它通过将 crystalInRing 除以每个 panel 中的晶体数量（GetCrystalNumYInPanel()）来确定晶体位于哪个 panel
    module1 = crystalInRing % GetCrystalNumYInPanel() / GetCrystalNumYInModule(); // 通过取余数来找到晶体相对于 panel 的位置，然后将结果除以每个 module 中的晶体数量
    block1 = crystalInRing % GetCrystalNumYInModule() / GetCrystalNumYInBlock();  // 通过取余数来找到晶体相对于 module 的位置，然后将结果除以每个 block 中的晶体数量
    cry1 = crystalInRing % GetCrystalNumYInBlock();                               // 通过取余数来找到晶体在 block 中的相对位置

    // 计算晶体在scanner上的极坐标角度 sita1
    double sita1;
    sita1 = double(panel1) * 2 * PI / panelNum;

    double panelOffset1; // 计算晶体在 panel 内的y方向偏移量
    panelOffset1 = module1 * modulePitch.y + (modulePitch.y - moduleSize.y) / 2 +
                   block1 * blockPitch.y + (blockPitch.y - blockSize.y) / 2 +
                   cry1 * crystalPitch.y + (crystalPitch.y - crystalSize.y) / 2 -
                   GetPanelSizeY() / 2;
    // 计算晶体的位置坐标
    crystalPos[0].x = scannerRadius * cos(sita1) + panelOffset1 * cos(sita1 + PI / 2);
    crystalPos[0].y = scannerRadius * sin(sita1) + panelOffset1 * sin(sita1 + PI / 2);
    crystalPos[3].x = crystalPos[0].x;
    crystalPos[3].y = crystalPos[0].y;

    crystalPos[1].x = scannerRadius * cos(sita1) + (panelOffset1 + crystalSize.y) * cos(sita1 + PI / 2);
    crystalPos[1].y = scannerRadius * sin(sita1) + (panelOffset1 + crystalSize.y) * sin(sita1 + PI / 2);
    crystalPos[2].x = crystalPos[1].x;
    crystalPos[2].y = crystalPos[1].y;
}

double UPETScanner::GetRingMinCoordinateZ(int ring) const
{
    // 异常处理
    // if (ring < 0 || ring > GetRingNum() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetRingMinCoordinateZ. Input parameter ring: %d out of range.\n", ring);
    // 	double z = 0.0; // 赋值，避免z值不可控
    // 	return z;
    // }
    double z = int(ring / GetCrystalNumZInModule()) * modulePitch.z + (modulePitch.z - moduleSize.z) / 2 +
               int(ring % GetCrystalNumZInModule() / GetCrystalNumZInBlock()) * blockPitch.z + (blockPitch.z - blockSize.z) / 2 +
               ring % GetCrystalNumZInBlock() * crystalPitch.z + (crystalPitch.z - crystalSize.z) / 2 -
               GetLengthZ() / 2;
    return z;
}

void UPETScanner::GetCrystalIDFromLORID(size_t LORID, int &crystalID1, int &crystalID2) const
{
    // 异常处理
    // if (LORID < 0 || LORID > GetLORNum() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetCrystalIDFromLORID. Input parameters out of range.\n");
    // 	return false;
    // }
    int crystalNumOneRing = GetCrystalNumOneRing();
    int binNum = GetBinNum();
    int viewNum = GetViewNum();
    int bin = LORID % binNum;
    int view = LORID / binNum % viewNum;
    int slice = LORID / (binNum * viewNum);
    int cry1, cry2, ring1, ring2;
    GetCrystalIDInRingFromViewBin(view, bin, cry1, cry2); // 由view, bin来获取环上晶体的索引
    GetRing1Ring2FromSlice(slice, ring1, ring2);          // 由slice来获取晶体环的索引
    crystalID1 = cry1 + ring1 * crystalNumOneRing;        // 映射到晶体global ID
    crystalID2 = cry2 + ring2 * crystalNumOneRing;        // 映射到晶体global ID
}

void UPETScanner::GetLORIDFromCrystalID(int crystalID1, int crystalID2, size_t &LORID) const
{
    // 异常处理
    // if (crystalID1 < 0 || crystalID1 > GetCrystalNum() - 1 || crystalID2 < 0 || crystalID2 > GetCrystalNum() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetLORIDFromCrystalID. Input parameters out of range.\n");
    // 	LORID = 0;
    // 	return false;
    // }
    int crystalNumOneRing = GetCrystalNumOneRing();
    int cry1 = crystalID1 % crystalNumOneRing;
    int cry2 = crystalID2 % crystalNumOneRing;
    int ring1 = crystalID1 / crystalNumOneRing;
    int ring2 = crystalID2 / crystalNumOneRing;
    GetLORIDFromRingAndCrystalInRing(ring1, cry1, ring2, cry2, LORID);
}

void UPETScanner::GetRing1Ring2FromSlice(int slice, int &ring1, int &ring2) const
{
    // 异常处理
    // if (slice < 0 || slice > GetSliceNum() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetRing1Ring2FromSlice. Input parameters out of range.\n");
    // 	ring1 = 0;
    // 	ring2 = 0;
    // 	return false;
    // }
    ring1 = slice / GetRingNum();
    ring2 = slice % GetRingNum();
}

void UPETScanner::GetCrystalIDInRingFromViewBin(int view, int bin, int &cry1, int &cry2) const
{
    // 异常处理
    // if (view < 0 || view > GetViewNum() - 1 || bin < 0 || bin > GetBinNum() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetCrystalIDInRingFromViewBin. Input parameters out of range.\n");
    // 	cry1 = 0;
    // 	cry2 = 0;
    // 	return false;
    // }
    int crystalOneRing = GetCrystalNumOneRing();
    cry2 = bin / 2 + 1;
    cry1 = crystalOneRing + (1 - bin % 2) - cry2;

    cry2 = (cry2 + view) % crystalOneRing;
    cry1 = (cry1 + view) % crystalOneRing;
}

void UPETScanner::GetLORIDFromRingAndCrystalInRing(int ring1, int cry1, int ring2, int cry2, size_t &LORID) const
{
    // 异常处理
    // if (ring1 < 0 || ring1 > GetRingNum() - 1 || ring2 < 0 || ring2 > GetRingNum() - 1 || cry1 < 0 || cry1 > GetCrystalNumOneRing() - 1 || cry2 < 0 || cry2 > GetCrystalNumOneRing() - 1)
    // {
    // 	printf("Error! At UPETScanner::GetLORIDFromRingAndCrystalInRing. Input parameters out of range.\n");
    // 	LORID = 0;
    // 	return false;
    // }
    int cry1t = cry1;
    int cry2t = cry2;
    size_t crystalOneRing = GetCrystalNumOneRing();
    size_t view = (cry1 + cry2) % crystalOneRing / 2;

    cry1 -= view;
    cry2 -= view;
    if (cry1 <= 0)
        cry1 += crystalOneRing; // 20180524:crystalOneRing / 2;
    if (cry2 <= 0)
        cry2 += crystalOneRing;                   // 20180524:crystalOneRing / 2;
    int ring1real, ring2real, cry1real, cry2real; // larger crystalId is index "1"
    if (cry1 > cry2)
    {
        cry1real = cry1;
        cry2real = cry2;
        ring1real = ring1;
        ring2real = ring2;
    }
    else
    {
        cry1real = cry2;
        cry2real = cry1;
        ring1real = ring2;
        ring2real = ring1;
    }
    size_t bin = (crystalOneRing - 1) - (cry1real - cry2real);
    size_t slice = ring1real * GetRingNum() + ring2real; // 20220630Updated: int -> size_t
    LORID = slice * (crystalOneRing - 1) * (crystalOneRing / 2) + view * (crystalOneRing - 1) + bin;
}

int UPETScanner::IsGoodPair(int panel1, int panel2, int minSectorDifference) const
{
    // 异常处理
    // if (panel1 < 0 || panel1 > GetPanelNum() - 1 || panel2 < 0 || panel2 > GetPanelNum() - 1)
    // {
    // 	printf("Error! At UPETScanner::IsGoodPair. Input parameters out of range.\n");
    // 	return false;
    // }
    return (abs(panel1 - panel2) >= minSectorDifference && panelNum - abs(panel1 - panel2) >= minSectorDifference) ? 1 : 0;
}
