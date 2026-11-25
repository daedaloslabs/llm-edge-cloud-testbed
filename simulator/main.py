"""
Single-file simulator for the radio side of the LLM edge-cloud paper.

The code is intentionally kept in one file so that the compute measurements can
be connected without changing the rest of the experiment. The development
compute model is only a fallback. Final paper runs should use the measured
edge/cloud model and --requireMeasurements.

Examples
--------
Quick installation check:
    python main.py --quick

Full experiment:
    python main.py

Final run after Michael connects the measurements:
    python main.py --requireMeasurements
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import computeModel  # noqa: E402  (Michael's measured edge/cloud model)


# Basic simulation settings. Values that are not measured are kept here so they
# are easy to audit before the final paper run.

@dataclass
class SimulationConfig:
    carrierGHz: float = 28.0
    channelBandwidthHz: float = 400e6
    subcarrierSpacingHz: float = 120e3
    totalPrbs: int = 264
    prbBandwidthHz: float = 12.0 * 120e3
    noiseFigureDb: float = 7.0
    implementationLossDb: float = 2.0

    areaWidthM: float = 1000.0
    areaHeightM: float = 1000.0
    ueHeightM: float = 1.5
    ueAntennas: int = 4
    ueMaxPowerDbm: float = 23.0
    ueReceivePowerW: float = 0.80

    mimoEfficiency: float = 0.82
    beamformingEfficiency: float = 0.68
    maxSpectralEfficiencyPerStream: float = 7.4
    interferenceBeamLossDb: float = 14.0

    simulationWindowS: float = 120.0
    requestRatePerUe: float = 0.010
    minimumRequests: int = 350

    edgeOneWayDelayS: float = 0.002
    cloudOneWayDelayS: float = 0.018
    radioSchedulingDelayS: float = 0.001

    promptBytesPerToken: float = 5.0
    outputBytesPerToken: float = 6.0
    messageOverheadBytes: float = 900.0

    ttftLimitS: float = 1.50
    completionLimitS: float = 8.00
    # "weighted" policy: score = E/Emin + latencyWeight * T/Tmin over the feasible candidates
    latencyWeight: float = 1.0
    riskQuantile: float = 0.90

    ulPowerControlP0Dbm: float = -82.0
    ulPowerControlAlpha: float = 0.80

    edgeTransportEnergyJPerBit: float = 0.4e-8
    cloudTransportEnergyJPerBit: float = 2.0e-8

    calibrationSamples: int = 6000
    outputRandomSigma: float = 0.45


config = SimulationConfig()


# Two macro, four micro and eight pico cells. The deployment is deterministic;
# channel realizations and user positions change with the seed.

tierSettings = {
    "macro": {
        "txPowerDbm": 46.0,
        "heightM": 25.0,
        "antennas": 64,
        "dynamicPowerW": 245.0,
        "backgroundMean": 0.44,
    },
    "micro": {
        "txPowerDbm": 33.0,
        "heightM": 10.0,
        "antennas": 32,
        "dynamicPowerW": 68.0,
        "backgroundMean": 0.35,
    },
    "pico": {
        "txPowerDbm": 27.0,
        "heightM": 6.0,
        "antennas": 16,
        "dynamicPowerW": 21.0,
        "backgroundMean": 0.26,
    },
}

policyOrder = [
    "proposed",
    "noUncertainty",
    "fixedAssociation",
    "minimumLatency",
    "allEdge",
    "allCloud",
    "oracleOutput",
]

policyLabels = {
    "proposed": "Proposed",
    "noUncertainty": "No uncertainty",
    "fixedAssociation": "Fixed association",
    "minimumLatency": "Minimum latency",
    "allEdge": "All edge",
    "allCloud": "All cloud",
    "oracleOutput": "Oracle output",
}

radioProfiles = [
    (8, 8),
    (16, 24),
    (32, 48),
]


# -----------------------------------------------------------------------------
# Michael's integration points
# -----------------------------------------------------------------------------


def michaelComputeEstimate(serverType, inputTokens, outputTokens, concurrency, quantization):
    """
    Connect the real edge/cloud model here.

    Return None while testing the network code. For the final paper run this
    function should return measured/interpolated values in the following form:

    {
        "prefillSeconds": float,
        "decodeSeconds": float,       # total decode time for outputTokens
        "powerWatts": float,
        "energyJoules": float
    }

    A tpotSeconds field can be returned instead of decodeSeconds. In that case
    decodeSeconds is calculated as tpotSeconds * outputTokens.

    Connected: set MEASUREMENTS=edge/summary.csv,cloud/summary.csv (and MODEL_NAME)
    to use common/computeModel.py. Unset -> None -> development fallback.
    """
    model = computeModel.load()
    if model is None:
        return None
    return model.estimate(serverType, inputTokens, outputTokens, concurrency, quantization)


def michaelServerCapacity(serverType):
    """
    Optional second hook for measured serving capacity.

    Return an integer if the real serving setup has a known concurrency cap.
    Return None to use the temporary development value.
    """
    model = computeModel.load()
    if model is None:
        return None
    return model.capacity.get(serverType)


fallbackUsed = False


def temporaryComputeEstimate(serverType, inputTokens, outputTokens, concurrency, quantization):
    """Development-only profile. These numbers are not paper measurements."""
    global fallbackUsed
    fallbackUsed = True

    concurrency = max(1, int(concurrency))
    inputTokens = max(1.0, float(inputTokens))
    outputTokens = max(1.0, float(outputTokens))

    if serverType == "edge":
        prefill = 0.050 + 0.000080 * inputTokens
        tpot = 0.0118
        basePower = 112.0
        concurrencyFactor = 1.0 + 0.032 * (concurrency - 1) ** 0.78
    else:
        prefill = 0.028 + 0.000040 * inputTokens
        tpot = 0.0059
        basePower = 292.0
        concurrencyFactor = 1.0 + 0.019 * (concurrency - 1) ** 0.75

    quantization = str(quantization).upper()
    if quantization == "INT8":
        prefill *= 1.08
        tpot *= 1.10
        basePower *= 1.06
    elif quantization in {"FP16", "BF16"}:
        prefill *= 1.18
        tpot *= 1.27
        basePower *= 1.16

    prefill *= concurrencyFactor
    decode = tpot * outputTokens * concurrencyFactor
    utilization = min(1.0, 0.48 + 0.060 * math.sqrt(concurrency))
    power = basePower * utilization
    energy = power * (prefill + decode)

    return {
        "prefillSeconds": prefill,
        "decodeSeconds": decode,
        "powerWatts": power,
        "energyJoules": energy,
    }


def computeEstimate(serverType, inputTokens, outputTokens, concurrency, quantization):
    measured = michaelComputeEstimate(
        serverType,
        int(inputTokens),
        float(outputTokens),
        int(concurrency),
        quantization,
    )

    if measured is None:
        measured = temporaryComputeEstimate(
            serverType,
            inputTokens,
            outputTokens,
            concurrency,
            quantization,
        )

    measured = dict(measured)

    if "decodeSeconds" not in measured and "tpotSeconds" in measured:
        measured["decodeSeconds"] = float(measured["tpotSeconds"]) * max(float(outputTokens), 1.0)

    required = {"prefillSeconds", "decodeSeconds", "powerWatts"}
    missing = required.difference(measured.keys())
    if missing:
        raise ValueError("Compute model is missing: " + ", ".join(sorted(missing)))

    measured["prefillSeconds"] = float(measured["prefillSeconds"])
    measured["decodeSeconds"] = float(measured["decodeSeconds"])
    measured["powerWatts"] = float(measured["powerWatts"])

    if "energyJoules" in measured:
        measured["energyJoules"] = float(measured["energyJoules"])
    else:
        measured["energyJoules"] = measured["powerWatts"] * (
            measured["prefillSeconds"] + measured["decodeSeconds"]
        )

    measured["tpotSeconds"] = measured["decodeSeconds"] / max(float(outputTokens), 1.0)
    measured["serviceSeconds"] = measured["prefillSeconds"] + measured["decodeSeconds"]
    return measured


def serverCapacity(serverType):
    measured = michaelServerCapacity(serverType)
    if measured is not None:
        return max(1, int(measured))
    return 24 if serverType == "edge" else 96


# -----------------------------------------------------------------------------
# Topology, traffic and output-length prediction
# -----------------------------------------------------------------------------


def buildTopology():
    macroPositions = [(250.0, 500.0), (750.0, 500.0)]
    microPositions = [
        (250.0, 250.0),
        (250.0, 750.0),
        (750.0, 250.0),
        (750.0, 750.0),
    ]
    picoPositions = [
        (125.0, 125.0),
        (125.0, 500.0),
        (125.0, 875.0),
        (500.0, 125.0),
        (500.0, 875.0),
        (875.0, 125.0),
        (875.0, 500.0),
        (875.0, 875.0),
    ]

    rows = []
    baseStationId = 0

    for tier, positions in (
        ("macro", macroPositions),
        ("micro", microPositions),
        ("pico", picoPositions),
    ):
        settings = tierSettings[tier]
        for xM, yM in positions:
            rows.append({
                "baseStationId": baseStationId,
                "tier": tier,
                "xM": xM,
                "yM": yM,
                "txPowerDbm": settings["txPowerDbm"],
                "heightM": settings["heightM"],
                "antennas": settings["antennas"],
                "dynamicPowerW": settings["dynamicPowerW"],
                "backgroundMean": settings["backgroundMean"],
            })
            baseStationId += 1

    return pd.DataFrame(rows)


def generateUserPositions(numberUsers, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "userId": np.arange(numberUsers, dtype=int),
        "xM": rng.uniform(0.0, config.areaWidthM, numberUsers),
        "yM": rng.uniform(0.0, config.areaHeightM, numberUsers),
    })


def outputMeanFromClass(inputTokens, serviceClass):
    base = np.where(serviceClass == 0, 96.0, np.where(serviceClass == 1, 192.0, 384.0))
    inputFactor = 1.0 + 0.000055 * np.maximum(np.asarray(inputTokens) - 512.0, 0.0)
    return base * inputFactor


def sampleRequestFeatures(numberRows, rng):
    inputOptions = np.array([256, 512, 1024, 2048, 4096])
    inputProbabilities = np.array([0.18, 0.27, 0.28, 0.19, 0.08])

    inputTokens = rng.choice(inputOptions, size=numberRows, p=inputProbabilities)
    serviceClass = rng.choice(np.array([0, 1, 2]), size=numberRows, p=np.array([0.45, 0.35, 0.20]))
    quantization = rng.choice(np.array(["INT4", "INT8"]), size=numberRows, p=np.array([0.75, 0.25]))

    meanOutput = outputMeanFromClass(inputTokens, serviceClass)
    sigma = config.outputRandomSigma
    logMean = np.log(meanOutput) - 0.5 * sigma * sigma
    actualOutput = rng.lognormal(logMean, sigma)
    actualOutput = np.clip(np.rint(actualOutput), 24, 768).astype(int)

    return pd.DataFrame({
        "inputTokens": inputTokens,
        "serviceClass": serviceClass,
        "quantization": quantization,
        "actualOutputTokens": actualOutput,
    })


def trainOutputPredictors(seed, quick=False):
    rng = np.random.default_rng(seed)
    sampleCount = 1200 if quick else config.calibrationSamples
    calibration = sampleRequestFeatures(sampleCount, rng)

    features = calibration[["inputTokens", "serviceClass"]].to_numpy(dtype=float)
    target = calibration["actualOutputTokens"].to_numpy(dtype=float)

    medianModel = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.50,
        n_estimators=35 if quick else 160,
        max_depth=2,
        learning_rate=0.05,
        random_state=seed,
    )
    riskModel = GradientBoostingRegressor(
        loss="quantile",
        alpha=config.riskQuantile,
        n_estimators=40 if quick else 180,
        max_depth=2,
        learning_rate=0.05,
        random_state=seed + 1,
    )

    medianModel.fit(features, target)
    riskModel.fit(features, target)
    return medianModel, riskModel


def generateRequests(users, seed, medianModel, riskModel):
    rng = np.random.default_rng(seed + 5000)
    rows = []
    requestId = 0

    meanRequests = config.requestRatePerUe * config.simulationWindowS

    for user in users.itertuples(index=False):
        count = int(rng.poisson(meanRequests))
        if count == 0:
            continue

        arrivals = np.sort(rng.uniform(0.0, config.simulationWindowS, count))
        features = sampleRequestFeatures(count, rng)

        for localIndex in range(count):
            row = features.iloc[localIndex]
            rows.append({
                "requestId": requestId,
                "userId": int(user.userId),
                "arrivalS": float(arrivals[localIndex]),
                "inputTokens": int(row.inputTokens),
                "serviceClass": int(row.serviceClass),
                "quantization": str(row.quantization),
                "actualOutputTokens": int(row.actualOutputTokens),
            })
            requestId += 1

    if len(rows) < config.minimumRequests:
        extraCount = config.minimumRequests - len(rows)
        chosenUsers = rng.integers(0, len(users), extraCount)
        arrivals = rng.uniform(0.0, config.simulationWindowS, extraCount)
        features = sampleRequestFeatures(extraCount, rng)

        for localIndex in range(extraCount):
            row = features.iloc[localIndex]
            rows.append({
                "requestId": requestId,
                "userId": int(chosenUsers[localIndex]),
                "arrivalS": float(arrivals[localIndex]),
                "inputTokens": int(row.inputTokens),
                "serviceClass": int(row.serviceClass),
                "quantization": str(row.quantization),
                "actualOutputTokens": int(row.actualOutputTokens),
            })
            requestId += 1

    requests = pd.DataFrame(rows).sort_values("arrivalS").reset_index(drop=True)
    x = requests[["inputTokens", "serviceClass"]].to_numpy(dtype=float)
    requests["medianOutputTokens"] = np.clip(medianModel.predict(x), 24, 768)
    requests["riskOutputTokens"] = np.clip(riskModel.predict(x), 24, 768)
    return requests


# -----------------------------------------------------------------------------
# 3GPP-style large scale channel and rank-aware MIMO abstraction
# -----------------------------------------------------------------------------


def losProbability(distance2dM, tier):
    distance2dM = np.maximum(distance2dM, 1.0)
    if tier == "macro":
        return (
            np.minimum(18.0 / distance2dM, 1.0)
            * (1.0 - np.exp(-distance2dM / 63.0))
            + np.exp(-distance2dM / 63.0)
        )

    return (
        np.minimum(18.0 / distance2dM, 1.0)
        * (1.0 - np.exp(-distance2dM / 36.0))
        + np.exp(-distance2dM / 36.0)
    )


def pathLossDb(distance3dM, tier, isLos):
    fc = config.carrierGHz
    distance3dM = np.maximum(distance3dM, 10.0)

    if tier == "macro":
        los = 28.0 + 22.0 * np.log10(distance3dM) + 20.0 * np.log10(fc)
        nlos = 13.54 + 39.08 * np.log10(distance3dM) + 20.0 * np.log10(fc)
        return np.where(isLos, los, np.maximum(los, nlos))

    los = 32.4 + 21.0 * np.log10(distance3dM) + 20.0 * np.log10(fc)
    nlos = 35.3 * np.log10(distance3dM) + 22.4 + 21.3 * np.log10(fc)
    return np.where(isLos, los, np.maximum(los, nlos))


def calculateLinkTable(users, baseStations, seed):
    rng = np.random.default_rng(seed + 10000)

    userX = users["xM"].to_numpy()[:, None]
    userY = users["yM"].to_numpy()[:, None]
    bsX = baseStations["xM"].to_numpy()[None, :]
    bsY = baseStations["yM"].to_numpy()[None, :]

    distance2d = np.sqrt((userX - bsX) ** 2 + (userY - bsY) ** 2)
    numberUsers, numberBaseStations = distance2d.shape

    pathLoss = np.zeros_like(distance2d)
    losMatrix = np.zeros_like(distance2d, dtype=bool)
    mimoRank = np.ones_like(distance2d, dtype=int)
    beamGainDb = np.zeros_like(distance2d)
    referenceRsrpDbm = np.zeros_like(distance2d)

    for bsIndex, bs in baseStations.iterrows():
        heightDifference = float(bs.heightM) - config.ueHeightM
        distance3d = np.sqrt(distance2d[:, bsIndex] ** 2 + heightDifference ** 2)

        probability = losProbability(distance2d[:, bsIndex], bs.tier)
        isLos = rng.random(numberUsers) < probability
        loss = pathLossDb(distance3d, bs.tier, isLos)

        shadowStd = np.where(isLos, 4.0, 6.0 if bs.tier == "macro" else 7.82)
        loss = loss + rng.normal(0.0, shadowStd)

        maxRank = min(config.ueAntennas, int(bs.antennas), 4)
        spatialProbability = np.where(isLos, 0.30, 0.62)
        rank = 1 + rng.binomial(maxRank - 1, spatialProbability)

        rawBeamGain = (
            config.beamformingEfficiency
            * float(bs.antennas)
            * config.ueAntennas
            / np.maximum(rank, 1)
        )
        gainDb = 10.0 * np.log10(np.maximum(rawBeamGain, 1.0))

        pathLoss[:, bsIndex] = loss
        losMatrix[:, bsIndex] = isLos
        mimoRank[:, bsIndex] = rank
        beamGainDb[:, bsIndex] = gainDb

        perPrbPowerDbm = float(bs.txPowerDbm) - 10.0 * math.log10(config.totalPrbs)
        referenceRsrpDbm[:, bsIndex] = perPrbPowerDbm + gainDb - loss - config.implementationLossDb

    interferenceReferenceMw = np.zeros_like(pathLoss)
    for bsIndex, bs in baseStations.iterrows():
        perPrbPowerDbm = float(bs.txPowerDbm) - 10.0 * math.log10(config.totalPrbs)
        interferenceDbm = (
            perPrbPowerDbm
            - pathLoss[:, int(bsIndex)]
            - config.interferenceBeamLossDb
        )
        interferenceReferenceMw[:, int(bsIndex)] = 10.0 ** (interferenceDbm / 10.0)

    return {
        "distance2dM": distance2d,
        "pathLossDb": pathLoss,
        "isLos": losMatrix,
        "mimoRank": mimoRank,
        "beamGainDb": beamGainDb,
        "referenceRsrpDbm": referenceRsrpDbm,
        "interferenceReferenceMw": interferenceReferenceMw,
    }


def mimoSpectralEfficiency(sinrDb, rank):
    rank = max(1, int(rank))
    sinrLinear = 10.0 ** (float(sinrDb) / 10.0)
    efficiency = config.mimoEfficiency * rank * math.log2(1.0 + sinrLinear / rank)
    return max(0.05, min(efficiency, config.maxSpectralEfficiencyPerStream * rank))


# -----------------------------------------------------------------------------
# Dynamic radio and compute states
# -----------------------------------------------------------------------------


class RadioState:
    def __init__(self, baseStations, seed):
        self.baseStations = baseStations
        self.seed = int(seed)
        self.reservations = {
            "ul": [[] for _ in range(len(baseStations))],
            "dl": [[] for _ in range(len(baseStations))],
        }

        rng = np.random.default_rng(seed + 18000)
        self.backgroundPhase = rng.uniform(0.0, 2.0 * math.pi, len(baseStations))
        self.backgroundJitter = rng.uniform(-0.04, 0.04, len(baseStations))

    def backgroundFraction(self, baseStationId, direction, timeS):
        bs = self.baseStations.iloc[int(baseStationId)]
        mean = float(bs.backgroundMean)
        phase = self.backgroundPhase[int(baseStationId)]
        wave = 0.10 * math.sin(2.0 * math.pi * timeS / 35.0 + phase)
        directionAdjustment = 0.05 if direction == "dl" else -0.04
        value = mean + wave + self.backgroundJitter[int(baseStationId)] + directionAdjustment
        return float(np.clip(value, 0.08, 0.82))

    def cleanup(self, timeS):
        for direction in ("ul", "dl"):
            for baseStationId in range(len(self.baseStations)):
                current = self.reservations[direction][baseStationId]
                self.reservations[direction][baseStationId] = [
                    item for item in current if item[1] > timeS
                ]

    def llmUsedPrbs(self, direction, baseStationId, timeS):
        return sum(
            item[2]
            for item in self.reservations[direction][int(baseStationId)]
            if item[0] <= timeS < item[1]
        )

    def utilization(self, direction, timeS):
        values = []
        for baseStationId in range(len(self.baseStations)):
            background = self.backgroundFraction(baseStationId, direction, timeS)
            llm = self.llmUsedPrbs(direction, baseStationId, timeS) / config.totalPrbs
            values.append(float(np.clip(background + llm, 0.0, 1.0)))
        return np.asarray(values)

    def availablePrbs(self, direction, baseStationId, timeS):
        background = int(round(self.backgroundFraction(baseStationId, direction, timeS) * config.totalPrbs))
        llm = self.llmUsedPrbs(direction, baseStationId, timeS)
        return max(1, config.totalPrbs - background - llm)

    def allocatePrbs(self, direction, baseStationId, timeS, requestedPrbs):
        available = self.availablePrbs(direction, baseStationId, timeS)
        return max(1, min(int(requestedPrbs), int(available)))

    def reserve(self, direction, baseStationId, startS, endS, prbs):
        if endS <= startS:
            return
        self.reservations[direction][int(baseStationId)].append(
            (float(startS), float(endS), int(prbs))
        )


class ComputeState:
    def __init__(self):
        self.finishTimes = {
            "edge": np.zeros(serverCapacity("edge"), dtype=float),
            "cloud": np.zeros(serverCapacity("cloud"), dtype=float),
        }

    def inspect(self, serverType, arrivalS):
        finish = self.finishTimes[serverType]
        active = int(np.sum(finish > arrivalS))
        free = np.where(finish <= arrivalS)[0]

        if len(free) > 0:
            slotIndex = int(free[0])
            waitS = 0.0
        else:
            slotIndex = int(np.argmin(finish))
            waitS = max(0.0, float(finish[slotIndex] - arrivalS))

        return active, waitS, slotIndex

    def commit(self, serverType, slotIndex, startS, serviceS):
        self.finishTimes[serverType][int(slotIndex)] = float(startS + serviceS)


# -----------------------------------------------------------------------------
# Radio link evaluation
# -----------------------------------------------------------------------------


def requestBits(inputTokens, outputTokens):
    inputBytes = config.messageOverheadBytes + config.promptBytesPerToken * float(inputTokens)
    outputBytes = config.messageOverheadBytes + config.outputBytesPerToken * float(outputTokens)
    return inputBytes * 8.0, outputBytes * 8.0


def noisePerPrbMw():
    noiseDbm = -174.0 + 10.0 * math.log10(config.prbBandwidthHz) + config.noiseFigureDb
    return 10.0 ** (noiseDbm / 10.0)


def downlinkSinrDb(userId, baseStationId, timeS, baseStations, linkTable, radioState):
    baseStationId = int(baseStationId)
    desiredDbm = float(linkTable["referenceRsrpDbm"][userId, baseStationId])
    desiredMw = 10.0 ** (desiredDbm / 10.0)

    utilization = radioState.utilization("dl", timeS)
    reference = linkTable["interferenceReferenceMw"][userId]
    interferenceMw = float(np.dot(utilization, reference))
    interferenceMw -= float(utilization[baseStationId] * reference[baseStationId])
    interferenceMw = max(0.0, interferenceMw)

    sinrLinear = desiredMw / (noisePerPrbMw() + interferenceMw)
    return 10.0 * math.log10(max(sinrLinear, 1e-12))

def uplinkPowerDbm(pathLossDb, prbCount):
    requested = (
        config.ulPowerControlP0Dbm
        + 10.0 * math.log10(max(prbCount, 1))
        + config.ulPowerControlAlpha * float(pathLossDb)
    )
    return float(np.clip(requested, -20.0, config.ueMaxPowerDbm))


def uplinkSinrDb(userId, baseStationId, timeS, prbCount, baseStations, linkTable, radioState):
    pathLoss = float(linkTable["pathLossDb"][userId, int(baseStationId)])
    txPowerDbm = uplinkPowerDbm(pathLoss, prbCount)
    perPrbPowerDbm = txPowerDbm - 10.0 * math.log10(max(prbCount, 1))

    desiredDbm = (
        perPrbPowerDbm
        + float(linkTable["beamGainDb"][userId, int(baseStationId)])
        - pathLoss
        - config.implementationLossDb
    )
    desiredMw = 10.0 ** (desiredDbm / 10.0)

    ulUtilization = radioState.utilization("ul", timeS)
    otherLoad = np.delete(ulUtilization, int(baseStationId))
    meanOtherLoad = float(np.mean(otherLoad)) if len(otherLoad) else 0.0

    # Analytical inter-cell UL noise rise. This term is later validated against
    # the selected 5G-LENA scenarios; it avoids pretending that we know the
    # exact instantaneous position and power of every interfering UE.
    interferenceRiseLinear = 1.0 + 3.5 * meanOtherLoad
    denominator = noisePerPrbMw() * interferenceRiseLinear
    sinrLinear = desiredMw / denominator

    return 10.0 * math.log10(max(sinrLinear, 1e-12)), txPowerDbm


def radioTransferEstimate(
    request,
    userId,
    baseStationId,
    outputTokens,
    requestedUlPrbs,
    requestedDlPrbs,
    baseStations,
    linkTable,
    radioState,
    earliestUlStart,
    earliestDlStart,
):
    ulPrbs = radioState.allocatePrbs("ul", baseStationId, earliestUlStart, requestedUlPrbs)
    dlPrbs = radioState.allocatePrbs("dl", baseStationId, earliestDlStart, requestedDlPrbs)

    ulSinrDb, ueTxPowerDbm = uplinkSinrDb(
        userId,
        baseStationId,
        earliestUlStart,
        ulPrbs,
        baseStations,
        linkTable,
        radioState,
    )
    dlSinrDb = downlinkSinrDb(
        userId,
        baseStationId,
        earliestDlStart,
        baseStations,
        linkTable,
        radioState,
    )

    rank = int(linkTable["mimoRank"][userId, int(baseStationId)])
    ulEfficiency = mimoSpectralEfficiency(ulSinrDb, rank)
    dlEfficiency = mimoSpectralEfficiency(dlSinrDb, rank)

    ulRateBps = max(1.0, ulPrbs * config.prbBandwidthHz * ulEfficiency)
    dlRateBps = max(1.0, dlPrbs * config.prbBandwidthHz * dlEfficiency)

    inputBits, outputBits = requestBits(request.inputTokens, outputTokens)
    ulDuration = inputBits / ulRateBps + config.radioSchedulingDelayS
    dlDuration = outputBits / dlRateBps + config.radioSchedulingDelayS

    firstTokenBits = (config.messageOverheadBytes + config.outputBytesPerToken) * 8.0
    firstTokenDl = firstTokenBits / dlRateBps + config.radioSchedulingDelayS

    ueTxPowerW = 10.0 ** ((ueTxPowerDbm - 30.0) / 10.0)
    ueEnergy = ueTxPowerW * ulDuration + config.ueReceivePowerW * dlDuration

    bs = baseStations.iloc[int(baseStationId)]
    ulFraction = ulPrbs / config.totalPrbs
    dlFraction = dlPrbs / config.totalPrbs
    ranEnergy = float(bs.dynamicPowerW) * (
        ulFraction * ulDuration + dlFraction * dlDuration
    )

    return {
        "ulPrbs": int(ulPrbs),
        "dlPrbs": int(dlPrbs),
        "ulSinrDb": float(ulSinrDb),
        "dlSinrDb": float(dlSinrDb),
        "mimoRank": rank,
        "ulRateBps": float(ulRateBps),
        "dlRateBps": float(dlRateBps),
        "ulDurationS": float(ulDuration),
        "dlDurationS": float(dlDuration),
        "firstTokenDlS": float(firstTokenDl),
        "ueTxPowerDbm": float(ueTxPowerDbm),
        "ueEnergyJ": float(ueEnergy),
        "ranEnergyJ": float(ranEnergy),
    }


# -----------------------------------------------------------------------------
# Joint radio-compute decision
# -----------------------------------------------------------------------------


def edgeDelayForTier(tier):
    if tier == "macro":
        return config.edgeOneWayDelayS
    if tier == "micro":
        return config.edgeOneWayDelayS + 0.001
    return config.edgeOneWayDelayS + 0.002


def transportValues(serverType, tier, inputTokens, outputTokens):
    inputBits, outputBits = requestBits(inputTokens, outputTokens)
    totalBits = inputBits + outputBits

    if serverType == "edge":
        oneWayDelay = edgeDelayForTier(tier)
        energy = totalBits * config.edgeTransportEnergyJPerBit
    else:
        oneWayDelay = config.cloudOneWayDelayS
        energy = totalBits * config.cloudTransportEnergyJPerBit

    return float(oneWayDelay), float(energy)


def decisionOutputTokens(request, policy):
    if policy in {"proposed", "fixedAssociation", "weighted"}:
        return float(request.riskOutputTokens)
    if policy == "oracleOutput":
        return float(request.actualOutputTokens)
    return float(request.medianOutputTokens)


def candidateBaseStations(userId, linkTable, count=4):
    quality = linkTable["referenceRsrpDbm"][int(userId)]
    count = min(int(count), len(quality))
    indices = np.argpartition(quality, -count)[-count:]
    return indices[np.argsort(quality[indices])[::-1]]


def evaluateCandidate(
    request,
    userId,
    baseStationId,
    serverType,
    outputTokens,
    ulPrbRequest,
    dlPrbRequest,
    baseStations,
    linkTable,
    radioState,
    computeState,
):
    arrivalS = float(request.arrivalS)
    tier = str(baseStations.iloc[int(baseStationId)].tier)

    # First pass only estimates the UL, because compute arrival depends on it.
    radioFirst = radioTransferEstimate(
        request,
        userId,
        baseStationId,
        outputTokens,
        ulPrbRequest,
        dlPrbRequest,
        baseStations,
        linkTable,
        radioState,
        arrivalS,
        arrivalS,
    )

    oneWayDelay, transportEnergy = transportValues(
        serverType,
        tier,
        request.inputTokens,
        outputTokens,
    )

    computeArrival = arrivalS + radioFirst["ulDurationS"] + oneWayDelay
    activeCount, queueWaitS, slotIndex = computeState.inspect(serverType, computeArrival)

    compute = computeEstimate(
        serverType,
        request.inputTokens,
        outputTokens,
        activeCount + 1,
        request.quantization,
    )

    computeStart = computeArrival + queueWaitS
    computeEnd = computeStart + compute["serviceSeconds"]
    dlStart = computeEnd + oneWayDelay

    # Second pass updates DL availability and interference at its actual start.
    radio = radioTransferEstimate(
        request,
        userId,
        baseStationId,
        outputTokens,
        ulPrbRequest,
        dlPrbRequest,
        baseStations,
        linkTable,
        radioState,
        arrivalS,
        dlStart,
    )

    ttft = (
        radio["ulDurationS"]
        + oneWayDelay
        + queueWaitS
        + compute["prefillSeconds"]
        + compute["tpotSeconds"]
        + oneWayDelay
        + radio["firstTokenDlS"]
    )

    completion = (
        radio["ulDurationS"]
        + oneWayDelay
        + queueWaitS
        + compute["serviceSeconds"]
        + oneWayDelay
        + radio["dlDurationS"]
    )

    totalEnergy = (
        radio["ueEnergyJ"]
        + radio["ranEnergyJ"]
        + transportEnergy
        + compute["energyJoules"]
    )

    return {
        "baseStationId": int(baseStationId),
        "serverType": serverType,
        "slotIndex": int(slotIndex),
        "computeArrivalS": float(computeArrival),
        "computeStartS": float(computeStart),
        "computeEndS": float(computeEnd),
        "dlStartS": float(dlStart),
        "oneWayDelayS": oneWayDelay,
        "transportEnergyJ": transportEnergy,
        "ttftS": float(ttft),
        "completionS": float(completion),
        "totalEnergyJ": float(totalEnergy),
        "radio": radio,
        "compute": compute,
    }


def selectDecision(request, baseStations, linkTable, radioState, computeState, policy):
    userId = int(request.userId)
    strongest = int(np.argmax(linkTable["referenceRsrpDbm"][userId]))
    outputTokens = decisionOutputTokens(request, policy)

    if policy in {"allEdge", "allCloud", "fixedAssociation"}:
        bsCandidates = np.array([strongest], dtype=int)
    else:
        bsCandidates = candidateBaseStations(userId, linkTable, count=3)

    if policy == "allEdge":
        serverCandidates = ["edge"]
    elif policy == "allCloud":
        serverCandidates = ["cloud"]
    else:
        serverCandidates = ["edge", "cloud"]

    candidates = []

    for baseStationId in bsCandidates:
        for serverType in serverCandidates:
            for ulPrbs, dlPrbs in radioProfiles:
                candidate = evaluateCandidate(
                    request,
                    userId,
                    int(baseStationId),
                    serverType,
                    outputTokens,
                    ulPrbs,
                    dlPrbs,
                    baseStations,
                    linkTable,
                    radioState,
                    computeState,
                )
                candidates.append(candidate)

    if policy == "minimumLatency":
        return min(candidates, key=lambda item: item["completionS"])

    feasible = [
        item
        for item in candidates
        if item["ttftS"] <= config.ttftLimitS
        and item["completionS"] <= config.completionLimitS
    ]

    if feasible and policy == "weighted":
        # Energy and latency normalised by their best feasible value, so both are
        # dimensionless and >= 1; latencyWeight = 0 reproduces "proposed".
        energyMin = min(item["totalEnergyJ"] for item in feasible)
        latencyMin = min(item["completionS"] for item in feasible)
        return min(
            feasible,
            key=lambda item: item["totalEnergyJ"] / energyMin
            + config.latencyWeight * item["completionS"] / latencyMin,
        )

    if feasible:
        return min(feasible, key=lambda item: item["totalEnergyJ"])

    def violationScore(item):
        ttftExcess = max(0.0, item["ttftS"] / config.ttftLimitS - 1.0)
        completionExcess = max(0.0, item["completionS"] / config.completionLimitS - 1.0)
        return 1000.0 * (ttftExcess + completionExcess) + item["totalEnergyJ"]

    return min(candidates, key=violationScore)


def realizeDecision(request, decision, baseStations, linkTable, radioState, computeState):
    userId = int(request.userId)
    baseStationId = int(decision["baseStationId"])
    serverType = decision["serverType"]
    tier = str(baseStations.iloc[baseStationId].tier)
    arrivalS = float(request.arrivalS)

    actualOutput = float(request.actualOutputTokens)
    oneWayDelay, transportEnergy = transportValues(
        serverType,
        tier,
        request.inputTokens,
        actualOutput,
    )

    requestedUlPrbs = int(decision["radio"]["ulPrbs"])
    requestedDlPrbs = int(decision["radio"]["dlPrbs"])

    initialRadio = radioTransferEstimate(
        request,
        userId,
        baseStationId,
        actualOutput,
        requestedUlPrbs,
        requestedDlPrbs,
        baseStations,
        linkTable,
        radioState,
        arrivalS,
        arrivalS,
    )

    computeArrival = arrivalS + initialRadio["ulDurationS"] + oneWayDelay
    activeCount, queueWaitS, slotIndex = computeState.inspect(serverType, computeArrival)

    compute = computeEstimate(
        serverType,
        request.inputTokens,
        actualOutput,
        activeCount + 1,
        request.quantization,
    )

    computeStart = computeArrival + queueWaitS
    computeEnd = computeStart + compute["serviceSeconds"]
    dlStart = computeEnd + oneWayDelay

    radio = radioTransferEstimate(
        request,
        userId,
        baseStationId,
        actualOutput,
        requestedUlPrbs,
        requestedDlPrbs,
        baseStations,
        linkTable,
        radioState,
        arrivalS,
        dlStart,
    )

    ttft = (
        radio["ulDurationS"]
        + oneWayDelay
        + queueWaitS
        + compute["prefillSeconds"]
        + compute["tpotSeconds"]
        + oneWayDelay
        + radio["firstTokenDlS"]
    )
    completion = (
        radio["ulDurationS"]
        + oneWayDelay
        + queueWaitS
        + compute["serviceSeconds"]
        + oneWayDelay
        + radio["dlDurationS"]
    )

    totalEnergy = (
        radio["ueEnergyJ"]
        + radio["ranEnergyJ"]
        + transportEnergy
        + compute["energyJoules"]
    )

    radioState.reserve(
        "ul",
        baseStationId,
        arrivalS,
        arrivalS + radio["ulDurationS"],
        radio["ulPrbs"],
    )
    radioState.reserve(
        "dl",
        baseStationId,
        dlStart,
        dlStart + radio["dlDurationS"],
        radio["dlPrbs"],
    )
    computeState.commit(serverType, slotIndex, computeStart, compute["serviceSeconds"])

    violation = ttft > config.ttftLimitS or completion > config.completionLimitS

    return {
        "requestId": int(request.requestId),
        "userId": userId,
        "arrivalS": arrivalS,
        "policy": "",
        "baseStationId": baseStationId,
        "baseStationTier": tier,
        "serverType": serverType,
        "inputTokens": int(request.inputTokens),
        "actualOutputTokens": int(request.actualOutputTokens),
        "medianOutputTokens": float(request.medianOutputTokens),
        "riskOutputTokens": float(request.riskOutputTokens),
        "mimoRank": int(radio["mimoRank"]),
        "ulPrbs": int(radio["ulPrbs"]),
        "dlPrbs": int(radio["dlPrbs"]),
        "ulSinrDb": float(radio["ulSinrDb"]),
        "dlSinrDb": float(radio["dlSinrDb"]),
        "ulRateMbps": float(radio["ulRateBps"] / 1e6),
        "dlRateMbps": float(radio["dlRateBps"] / 1e6),
        "queueWaitS": float(queueWaitS),
        "ttftS": float(ttft),
        "completionS": float(completion),
        "ueEnergyJ": float(radio["ueEnergyJ"]),
        "ranEnergyJ": float(radio["ranEnergyJ"]),
        "transportEnergyJ": float(transportEnergy),
        "computeEnergyJ": float(compute["energyJoules"]),
        "totalEnergyJ": float(totalEnergy),
        "slaViolation": bool(violation),
    }


def runPolicy(requests, baseStations, linkTable, policy, seed):
    radioState = RadioState(baseStations, seed)
    computeState = ComputeState()
    rows = []

    for request in requests.itertuples(index=False):
        radioState.cleanup(float(request.arrivalS))
        decision = selectDecision(
            request,
            baseStations,
            linkTable,
            radioState,
            computeState,
            policy,
        )
        result = realizeDecision(
            request,
            decision,
            baseStations,
            linkTable,
            radioState,
            computeState,
        )
        result["policy"] = policy
        rows.append(result)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Experiments and statistics
# -----------------------------------------------------------------------------


def p95(values):
    return float(np.percentile(np.asarray(values, dtype=float), 95))


def summarizeRun(results, numberUsers, seed, policy):
    tierShare = results["baseStationTier"].value_counts(normalize=True)

    return {
        "numberUsers": int(numberUsers),
        "seed": int(seed),
        "policy": policy,
        "requestCount": int(len(results)),
        "meanTotalEnergyJ": float(results["totalEnergyJ"].mean()),
        "p95TtftS": p95(results["ttftS"]),
        "p95CompletionS": p95(results["completionS"]),
        "meanTtftS": float(results["ttftS"].mean()),
        "meanCompletionS": float(results["completionS"].mean()),
        "slaViolationRate": float(results["slaViolation"].mean()),
        "edgeRatio": float(np.mean(results["serverType"] == "edge")),
        "macroRatio": float(tierShare.get("macro", 0.0)),
        "microRatio": float(tierShare.get("micro", 0.0)),
        "picoRatio": float(tierShare.get("pico", 0.0)),
        "meanUeEnergyJ": float(results["ueEnergyJ"].mean()),
        "meanRanEnergyJ": float(results["ranEnergyJ"].mean()),
        "meanTransportEnergyJ": float(results["transportEnergyJ"].mean()),
        "meanComputeEnergyJ": float(results["computeEnergyJ"].mean()),
    }


def confidence95(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


def aggregateRuns(summaryFrame):
    metrics = [
        "meanTotalEnergyJ",
        "p95TtftS",
        "p95CompletionS",
        "slaViolationRate",
        "edgeRatio",
        "macroRatio",
        "microRatio",
        "picoRatio",
        "meanUeEnergyJ",
        "meanRanEnergyJ",
        "meanTransportEnergyJ",
        "meanComputeEnergyJ",
    ]

    rows = []
    for (numberUsers, policy), group in summaryFrame.groupby(["numberUsers", "policy"]):
        row = {
            "numberUsers": int(numberUsers),
            "policy": policy,
            "seedCount": int(group["seed"].nunique()),
            "meanRequestCount": float(group["requestCount"].mean()),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[metric + "Ci95"] = confidence95(group[metric].to_numpy())
        rows.append(row)

    return pd.DataFrame(rows)


def runMainExperiment(userCounts, seeds, outputFolder, quick):
    baseStations = buildTopology()
    medianModel, riskModel = trainOutputPredictors(913, quick=quick)

    summaries = []
    sampleFrames = []

    for numberUsers in userCounts:
        for seed in seeds:
            print(f"Users={numberUsers}, seed={seed}")
            users = generateUserPositions(numberUsers, seed)
            requests = generateRequests(users, seed, medianModel, riskModel)
            linkTable = calculateLinkTable(users, baseStations, seed)

            for policy in policyOrder:
                results = runPolicy(requests, baseStations, linkTable, policy, seed)
                summaries.append(summarizeRun(results, numberUsers, seed, policy))

                if numberUsers == max(userCounts) and seed == seeds[0] and policy in {
                    "proposed",
                    "fixedAssociation",
                    "allEdge",
                    "allCloud",
                }:
                    sample = results.copy()
                    sample["numberUsers"] = numberUsers
                    sample["seed"] = seed
                    sampleFrames.append(sample)

    summaryFrame = pd.DataFrame(summaries)
    aggregateFrame = aggregateRuns(summaryFrame)

    summaryFrame.to_csv(outputFolder / "mainExperimentSummary.csv", index=False)
    aggregateFrame.to_csv(outputFolder / "mainExperimentAggregate.csv", index=False)

    if sampleFrames:
        pd.concat(sampleFrames, ignore_index=True).to_csv(
            outputFolder / "requestLevelExample.csv",
            index=False,
        )

    return summaryFrame, aggregateFrame


def runUncertaintyExperiment(numberUsers, seeds, outputFolder, quick):
    baseStations = buildTopology()
    uncertaintyValues = [0.25, 0.45, 0.65, 0.85] if not quick else [0.30, 0.70]
    rows = []

    originalSigma = config.outputRandomSigma

    try:
        for sigma in uncertaintyValues:
            config.outputRandomSigma = float(sigma)
            medianModel, riskModel = trainOutputPredictors(1200 + int(100 * sigma), quick=quick)

            for seed in seeds:
                users = generateUserPositions(numberUsers, seed)
                requests = generateRequests(users, seed + 300, medianModel, riskModel)
                linkTable = calculateLinkTable(users, baseStations, seed)

                for policy in ["proposed", "noUncertainty"]:
                    results = runPolicy(requests, baseStations, linkTable, policy, seed)
                    summary = summarizeRun(results, numberUsers, seed, policy)
                    summary["outputSigma"] = sigma
                    rows.append(summary)
    finally:
        config.outputRandomSigma = originalSigma

    frame = pd.DataFrame(rows)
    aggregate = (
        frame.groupby(["outputSigma", "policy"], as_index=False)
        .agg(
            slaViolationRate=("slaViolationRate", "mean"),
            meanTotalEnergyJ=("meanTotalEnergyJ", "mean"),
            p95CompletionS=("p95CompletionS", "mean"),
        )
    )
    aggregate.to_csv(outputFolder / "uncertaintyExperiment.csv", index=False)
    return aggregate


# -----------------------------------------------------------------------------
# Paper figures
# -----------------------------------------------------------------------------


def groupedBars(aggregate, metric, ciMetric, ylabel, title, policies, outputPath):
    userCounts = sorted(aggregate["numberUsers"].unique())
    x = np.arange(len(userCounts), dtype=float)
    width = min(0.14, 0.78 / max(1, len(policies)))

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    for index, policy in enumerate(policies):
        subset = aggregate[aggregate["policy"] == policy].set_index("numberUsers")
        values = [subset.loc[count, metric] for count in userCounts]
        errors = [subset.loc[count, ciMetric] for count in userCounts]
        offset = (index - (len(policies) - 1) / 2.0) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            label=policyLabels[policy],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in userCounts])
    ax.set_xlabel("Number of UEs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outputPath, dpi=240)
    plt.close(fig)


def plotCellAssociation(aggregate, outputPath):
    userCount = int(aggregate["numberUsers"].max())
    policies = ["proposed", "fixedAssociation"]
    subset = aggregate[
        (aggregate["numberUsers"] == userCount)
        & (aggregate["policy"].isin(policies))
    ].copy()
    subset["order"] = subset["policy"].apply(policies.index)
    subset = subset.sort_values("order")

    x = np.arange(len(subset))
    macro = subset["macroRatio"].to_numpy()
    micro = subset["microRatio"].to_numpy()
    pico = subset["picoRatio"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.bar(x, macro, label="Macro")
    ax.bar(x, micro, bottom=macro, label="Micro")
    ax.bar(x, pico, bottom=macro + micro, label="Pico")
    ax.set_xticks(x)
    ax.set_xticklabels([policyLabels[p] for p in subset["policy"]])
    ax.set_ylabel("Association ratio")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Cell-tier association, {userCount} UEs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputPath, dpi=240)
    plt.close(fig)


def plotUncertainty(uncertaintyFrame, outputPath):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for policy in ["proposed", "noUncertainty"]:
        subset = uncertaintyFrame[uncertaintyFrame["policy"] == policy].sort_values("outputSigma")
        ax.plot(
            subset["outputSigma"],
            subset["slaViolationRate"],
            marker="o",
            label=policyLabels[policy],
        )

    ax.set_xlabel("Output-length uncertainty")
    ax.set_ylabel("SLA violation rate")
    ax.set_title("Impact of output-length uncertainty")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputPath, dpi=240)
    plt.close(fig)


def plotEnergyBreakdown(aggregate, outputPath):
    userCount = int(aggregate["numberUsers"].max())
    policies = ["proposed", "fixedAssociation", "allEdge", "allCloud"]
    subset = aggregate[
        (aggregate["numberUsers"] == userCount)
        & (aggregate["policy"].isin(policies))
    ].copy()
    subset["order"] = subset["policy"].apply(policies.index)
    subset = subset.sort_values("order")

    x = np.arange(len(subset))
    ue = subset["meanUeEnergyJ"].to_numpy()
    ran = subset["meanRanEnergyJ"].to_numpy()
    transport = subset["meanTransportEnergyJ"].to_numpy()
    compute = subset["meanComputeEnergyJ"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.bar(x, ue, label="UE")
    ax.bar(x, ran, bottom=ue, label="RAN")
    ax.bar(x, transport, bottom=ue + ran, label="Transport")
    ax.bar(x, compute, bottom=ue + ran + transport, label="Compute")
    ax.set_xticks(x)
    ax.set_xticklabels([policyLabels[p] for p in subset["policy"]])
    ax.set_ylabel("Mean energy per request (J)")
    ax.set_title(f"End-to-end energy breakdown, {userCount} UEs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputPath, dpi=240)
    plt.close(fig)


def createFigures(aggregate, uncertaintyFrame, outputFolder):
    groupedBars(
        aggregate,
        "meanTotalEnergyJ",
        "meanTotalEnergyJCi95",
        "Mean energy per request (J)",
        "Total energy per request",
        policyOrder,
        outputFolder / "figure1TotalEnergy.png",
    )

    groupedBars(
        aggregate,
        "p95TtftS",
        "p95TtftSCi95",
        "p95 TTFT (s)",
        "Tail Time to First Token",
        policyOrder,
        outputFolder / "figure2P95Ttft.png",
    )

    groupedBars(
        aggregate,
        "p95CompletionS",
        "p95CompletionSCi95",
        "p95 completion latency (s)",
        "Tail completion latency",
        policyOrder,
        outputFolder / "figure3P95Completion.png",
    )

    groupedBars(
        aggregate,
        "edgeRatio",
        "edgeRatioCi95",
        "Edge execution ratio",
        "Edge-cloud placement",
        ["proposed", "noUncertainty", "fixedAssociation", "minimumLatency"],
        outputFolder / "figure4EdgeRatio.png",
    )

    plotCellAssociation(
        aggregate,
        outputFolder / "figure5CellTierAssociation.png",
    )

    plotUncertainty(
        uncertaintyFrame,
        outputFolder / "figure6UncertaintyAblation.png",
    )

    plotEnergyBreakdown(
        aggregate,
        outputFolder / "figure7EnergyBreakdown.png",
    )


# -----------------------------------------------------------------------------
# Sanity checks and command line
# -----------------------------------------------------------------------------


def validateResults(summaryFrame, aggregateFrame):
    if summaryFrame.empty or aggregateFrame.empty:
        raise RuntimeError("Simulation returned no results.")

    numeric = summaryFrame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("NaN or infinite value found in the results.")

    ratios = aggregateFrame[["macroRatio", "microRatio", "picoRatio"]].sum(axis=1)
    if np.any(np.abs(ratios - 1.0) > 1e-6):
        raise RuntimeError("Cell-tier association ratios do not sum to one.")

    for column in ["slaViolationRate", "edgeRatio", "macroRatio", "microRatio", "picoRatio"]:
        if np.any((aggregateFrame[column] < -1e-9) | (aggregateFrame[column] > 1.0 + 1e-9)):
            raise RuntimeError(f"Invalid probability in {column}.")


def writeMetadata(outputFolder, userCounts, seeds, quick):
    metadata = {
        "carrierGHz": config.carrierGHz,
        "channelBandwidthHz": config.channelBandwidthHz,
        "subcarrierSpacingHz": config.subcarrierSpacingHz,
        "totalPrbs": config.totalPrbs,
        "userCounts": list(userCounts),
        "seeds": list(seeds),
        "quickMode": bool(quick),
        "computeSource": "temporary fallback" if fallbackUsed else "Michael measured model",
        "radioModel": "3GPP-style UMa/UMi large-scale channel with rank-aware analytical MIMO",
        "notes": [
            "Final paper results must be rerun with the measured compute model.",
            "Selected radio operating points should be cross-checked in 5G-LENA.",
            "RAN and transport energy parameters must be cited or sensitivity-tested in the paper.",
        ],
    }

    with open(outputFolder / "runMetadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Small run used only to verify the installation.")
    parser.add_argument("--requireMeasurements", action="store_true", help="Stop if the temporary compute model is used.")
    parser.add_argument("--output", default="results", help="Directory used for CSV files and figures.")
    return parser.parse_args()


def main():
    global fallbackUsed

    args = parseArguments()
    outputFolder = Path(args.output)
    outputFolder.mkdir(parents=True, exist_ok=True)

    if args.quick:
        userCounts = [8, 16]
        seeds = [7]
        uncertaintySeeds = [7]
        config.minimumRequests = 10
    else:
        userCounts = [1000, 2000]
        seeds = [7, 19, 31, 43, 59, 71, 83, 97, 109, 127]
        uncertaintySeeds = seeds[:5]

    fallbackUsed = False

    if args.requireMeasurements:
        edgeProbe = michaelComputeEstimate("edge", 512, 128, 1, "INT4")
        cloudProbe = michaelComputeEstimate("cloud", 512, 128, 1, "INT4")
        if edgeProbe is None or cloudProbe is None:
            raise RuntimeError(
                "--requireMeasurements was used, but Michael's edge/cloud compute model is not connected."
            )

    print("Running the main experiment")
    summaryFrame, aggregateFrame = runMainExperiment(
        userCounts,
        seeds,
        outputFolder,
        args.quick,
    )

    print("Running the uncertainty ablation")
    uncertaintyFrame = runUncertaintyExperiment(
        max(userCounts),
        uncertaintySeeds,
        outputFolder,
        args.quick,
    )

    validateResults(summaryFrame, aggregateFrame)
    createFigures(aggregateFrame, uncertaintyFrame, outputFolder)
    writeMetadata(outputFolder, userCounts, seeds, args.quick)

    if args.requireMeasurements and fallbackUsed:
        raise RuntimeError(
            "The temporary compute model was used. Connect Michael's measured model before the final paper run."
        )

    if fallbackUsed:
        print("WARNING: temporary compute numbers were used. Do not report these values in the paper.")
    else:
        print("Measured compute model detected.")

    print("Done. Results:", outputFolder.resolve())


if __name__ == "__main__":
    main()
