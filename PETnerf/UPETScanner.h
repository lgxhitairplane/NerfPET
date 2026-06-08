/**
 * @file      UPETScanner.h
 * @brief     The header file for the UPETScanner class.
 * @author    Ang Li
 * @date      2019-6-8
 * @copyright (C) RAYSOLUTION Co., Ltd. All rights reserved. This software and its associated documentation files are the exclusive property of RAYSOLUTION Co., Ltd. The original development of this software was carried out by PETLab. PETLab retains the right to perform subsequent development on this software for RAYSOLUTION Co., Ltd. under the terms and conditions of the agreement between the two parties.
 * @attention
 * Edit History:
 * 2019-6-8; Ang Li; Create.
 */
#ifndef __PET_SCANNER_H
#define __PET_SCANNER_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include "UVolume.h"

/**
 * @class UPETScanner
 * @brief This class represents a PET scanner and provides various methods to set and get its properties.
 */
class UPETScanner
{
public:
    UPETScanner(void) = default;
    virtual ~UPETScanner(void) = default;
    UPETScanner(const UPETScanner &) = default;
    UPETScanner &operator=(const UPETScanner &) = default;
    // set&get
    void SetCrystalNum(int numX, int numY, int numZ);
    void SetBlockNum(int numX, int numY, int numZ);
    void SetModuleNum(int numX, int numY, int numZ);
    void SetCrystalSize(double sizeX, double sizeY, double sizeZ);
    void SetBlockSize(double sizeX, double sizeY, double sizeZ);
    void SetModuleSize(double sizeX, double sizeY, double sizeZ);
    void SetPanelSize(double sizeX, double sizeY, double sizeZ);
    void SetCrystalPitch(double pitchX, double pitchY, double pitchZ);
    void SetBlockPitch(double pitchX, double pitchY, double pitchZ);
    void SetModulePitch(double pitchX, double pitchY, double pitchZ);
    void SetRingPara(int nPanelNum, int nCrystalClockwiseOffset, double fScannerRadius);

    int GetCrystalNumOneRing() const;
    int GetRingNum() const;
    int GetBlockNumOneRing() const;
    int GetBlockRingNum() const;
    int GetBinNum() const;
    int GetViewNum() const;
    int GetSliceNum() const;
    size_t GetLORNum() const; // 20220629Updated: int -> size_t
    int GetCrystalNum() const;
    int GetBlockNum() const;
    int GetCrystalNumZInPanel() const;
    int GetCrystalNumYInPanel() const;
    int GetCrystalNumZInModule() const;
    int GetCrystalNumYInModule() const;
    int GetCrystalNumZInBlock() const;
    int GetCrystalNumYInBlock() const;
    int GetBlockNumZInPanel() const;
    int GetBlockNumYInPanel() const;
    int GetBlockNumZInModule() const;
    int GetBlockNumYInModule() const;
    int GetModuleNumZInPanel() const;
    int GetModuleNumYInPanel() const;
    int GetPanelNum() const;
    double GetLengthZ() const;
    double GetPanelSizeZ() const;
    double GetPanelSizeY() const;
    double GetModulePitchZ() const;
    double GetModulePitchY() const;
    double GetModuleSizeZ() const;
    double GetModuleSizeY() const;
    double GetBlockPitchZ() const;
    double GetBlockPitchY() const;
    double GetBlockSizeZ() const;
    double GetBlockSizeY() const;
    double GetCrystalPitchY() const;
    double GetCrystalPitchZ() const;
    double GetCrystalSizeY() const;
    double GetCrystalSizeZ() const;
    double GetRadius() const;
    int GetBlockInRingFromCrystalId(int crystalId) const;
    int GetCrystalClockwiseOffset() const;
    /**
     * @brief Get the 4 vertices positions of one crystal.
     * @param crystalInRing Crystal index in ring. At the range of [0, crystalNumOneRing - 1]
     * @param ring Ring index of this crystal. At the range of [0, ringNum - 1]
     * @param crystalPos Get the crystal position which has 4 elements: UVolume3D<double> crystalPos[4];
     * @return true if successful, false otherwise.
     */
    void GetCrystalPosition(int crystalInRing, int ring, UVolume3D<double> *crystal) const;
    /**
     * @brief Get the minimum z coordinate of one ring.
     * @param ring Ring index of this crystal. At the range of [0, ringNum - 1]
     * @return Minimum z coordinate of the ring.
     */
    double GetRingMinCoordinateZ(int ring) const;
    /**
     * @brief Convert LORID to global crystal ID.
     * @param LORID LOR index. At the range of [0, LORNum - 1]
     * @param crystalID1 Get the global crystal1 index. At the range of [0, crystalNum - 1]
     * @param crystalID2 Get the global crystal2 index. At the range of [0, crystalNum - 1]
     * @return true if successful, false otherwise.
     */
    void GetCrystalIDFromLORID(size_t LORID, int &crystalID1, int &crystalID2) const;
    /**
     * @brief Convert global crystal ID to LOR ID.
     * @param crystalID1 Get the global crystal1 index. At the range of [0, crystalNum - 1]
     * @param crystalID2 Get the global crystal2 index. At the range of [0, crystalNum - 1]
     * @param LORID Get the LOR index. At the range of [0, LORNum - 1]
     * @return true if successful, false otherwise.
     */
    void GetLORIDFromCrystalID(int crystalID1, int crystalID2, size_t &LORID) const; // 20220329Updated: int LORID -> size_t LORID
    /**
     * @brief Convert LORID to crystal ID. Slice to ring1 and ring2.
     * @param slice Slice index of this LOR.
     * @param ring1 Get the ring1 index. At the range of [0, ringNum - 1]
     * @param ring2 Get the ring2 index. At the range of [0, ringNum - 1]
     * @return true if successful, false otherwise.
     */
    void GetRing1Ring2FromSlice(int slice, int &ring1, int &ring2) const;
    /**
     * @brief Convert LORID to crystal ID. view, bin to crystal1, crystal2.
     * @param view View index of this LOR.
     * @param bin Bin index of this LOR.
     * @param cry1 Get the index of crystal1 on the ring. At the range of [0, crystalOneRing - 1]
     * @param cry2 Get the index of crystal2 on the ring. At the range of [0, crystalOneRing - 1]
     * @return true if successful, false otherwise.
     */
    void GetCrystalIDInRingFromViewBin(int view, int bin, int &cry1, int &cry2) const;
    /**
     * @brief Convert Crystal ID to LOR ID.
     * @param ring1 The ring1 index. At the range of [0, ringNum - 1]
     * @param cry1 The index of crystal1 on the ring. At the range of [0, crystalOneRing - 1]
     * @param ring2 The ring2 index. At the range of [0, ringNum - 1]
     * @param cry2 The index of crystal2 on the ring. At the range of [0, crystalOneRing - 1]
     * @param LORID Get the LOR index.
     * @return true if successful, false otherwise.
     */
    void GetLORIDFromRingAndCrystalInRing(int ring1, int cry1, int ring2, int cry2, size_t &LORID) const;
    /**
     * @brief Determine if the 2 panels have a large distance.
     * @param panel1 The panel1 index.
     * @param panel2 The panel2 index.
     * @param minSectorDifference The minimum difference between the 2 panels.
     * @return true if successful, false otherwise.
     */
    int IsGoodPair(int panel1, int panel2, int minSectorDifference) const;

public:
    void InitD80Scanner();
    void InitD180Scanner();
    void InitE180Scanner();
    void InitBrainScanner();
    void InitDPET100Scanner();
    void InitDPET200Scanner();
    void InitDigitMI930();
    void InitDigitMI925();
    void InitDigitMI920();
    void InitDigitMI930_24();
    void InitDigitMIi30();
    virtual void Init() = delete; ///< An interface for subclasses to configure the system structure.

protected:
    UVolume3D<double> crystalSize;	///< Size of the crystal.
    UVolume3D<double> crystalPitch; ///< Pitch of the crystal.
    UVolume3D<int> crystalNum;		///< Number of crystals.

    UVolume3D<double> blockSize;  ///< Size of the block.
    UVolume3D<double> blockPitch; ///< Pitch of the block.
    UVolume3D<int> blockNum;	  ///< Number of blocks.

    UVolume3D<double> moduleSize;  ///< Size of the module.
    UVolume3D<double> modulePitch; ///< Pitch of the module.
    UVolume3D<int> moduleNum;	   ///< Number of modules.

    UVolume3D<double> panelSize;  ///< Size of the panel.
    UVolume3D<double> panelPitch; ///< Pitch of the panel.
    int panelNum;				  ///< Number of panels.
    int crystalClockwiseOffset;	  ///< Offset of the crystal in a clockwise direction.
    double scannerRadius;		  ///< Radius of the scanner.
};

#endif