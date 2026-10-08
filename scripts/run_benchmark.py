#!/usr/bin/env python3

import sys
import os
import time
import json
import argparse

from pyworkflow.project import Manager
from pyworkflow.utils.path import getBaseName, cleanPath
from pyworkflow.protocol import getProtocolFromDb

from pwem.protocols import ProtImportPdb
from pwem.objects import Boolean
from pwchem.protocols import ProtChemPrepareReceptor

from openstructure.protocols import ProtCompareStructures
from chimera.protocols.protocol_discrepancies import ChimeraProtDiscrepancies

from biofold.protocols import (
    ProtBoltz,
    ProtChai,
    ProtProtenix,
    ProtIntelliFold,
    ProtImportPredictions
)


# ============================================================================
# Protocol labels
# ============================================================================

PROT_LABELS = [
    # Imports
    "Import reference {}",
    "Prepare reference {}",

    # Predictors
    "Boltz {}",
    "Chai {}",
    "Protenix {}",
    "IntelliFold {}",
    "Import AF3 {}",

    # Validation
    "OpenStructure Boltz {}",
    "OpenStructure Chai {}",
    "OpenStructure Protenix {}",
    "OpenStructure IntelliFold {}",
    "OpenStructure AF3 {}",

    # Residue-level RMSD
    "Chimera discrepancies {}",
]


OST_LABELS = {
    "Boltz": 7,
    "Chai": 8,
    "Protenix": 9,
    "IntelliFold": 10,
    "AF3": 11,
}


EXPECTED_CATEGORIES = [
    "proteinProtein",
    "proteinAntibody",
    "proteinDNA",
    "proteinRNA",
    "proteinPeptide",
]


# ============================================================================
# Benchmark case loading
# ============================================================================

def loadCases(fasta_dir, af3_dir, category=None, ids_file=None):
    """
    Load benchmark cases from the FASTA directory.

    fasta_dir:
        Root directory containing one directory per category.

    af3_dir:
        Root directory containing one directory per category with
        AF3 prediction ZIP files.

    category:
        Optional category to process.

    ids_file:
        Optional text file containing the PDB IDs to process.
        If omitted, all FASTA files in the selected category are used.
    """

    fasta_dir = os.path.abspath(fasta_dir)
    af3_dir = os.path.abspath(af3_dir)

    selected_ids = None

    if ids_file is not None:

        with open(ids_file) as f:
            selected_ids = {
                line.strip().upper()
                for line in f
                if line.strip()
                and not line.startswith("#")
            }

    cases = []

    if category is not None:

        categories = [category]

    else:

        categories = sorted(
            os.listdir(fasta_dir)
        )

    for category_path in categories:

        category_dir = os.path.join(
            fasta_dir,
            category_path
        )

        if not os.path.isdir(category_dir):
            continue

        for fasta_name in sorted(
            os.listdir(category_dir)
        ):

            if not fasta_name.lower().endswith(
                (".fasta", ".fa", ".faa")
            ):
                continue

            pdbId = os.path.splitext(
                fasta_name
            )[0].upper()

            if (
                selected_ids is not None
                and pdbId not in selected_ids
            ):
                continue

            case = {
                "id": pdbId,
                "category": category_path,
                "fasta": os.path.join(
                    category_dir,
                    fasta_name
                ),
                "af3": os.path.join(
                    af3_dir,
                    category_path,
                    f"{pdbId.lower()}.zip"
                )
            }

            cases.append(case)

    return cases


# ============================================================================
# Logging
# ============================================================================

def log_case(log_file, pdb_id, category, status, message=""):
    """Append benchmark status to a log file."""

    with open(log_file, "a") as f:
        f.write(
            f"{pdb_id}\t"
            f"{category}\t"
            f"{status}\t"
            f"{message}\n"
        )


# ============================================================================
# Slurm
# ============================================================================

def sendToSlurm(
    prot,
    memory=8192,
    hours=48,
    GPU=False,
    nGPUs=1,
    nMPIs=None,
    nThreads=None
):

    prot._useQueue.set(
        Boolean(True)
    )

    QUEUE_PARAMS = (
        "diritroncos",
        {
            "JOB_MEMORY": memory,
            "JOB_TIME": hours,
            "GPU_COUNT": nGPUs if GPU else 0,
            "JOB_NODES": (
                nMPIs
                if nMPIs
                else int(prot.numberOfMpi)
            ),
            "JOB_THREADS": (
                nThreads
                if nThreads
                else int(prot.numberOfThreads)
            ),
        }
    )

    prot._queueParams.set(
        json.dumps(QUEUE_PARAMS)
    )


# ============================================================================
# Scipion project helpers
# ============================================================================

def loadRunDic(proj, finished=True):
    """
    Return the protocols in the project as:

        {pdbId: {protLabel: prot}}
    """

    rDic = {}

    for prot in proj.getRuns():

        protLabel = prot.getObjLabel()
        pdbId = protLabel.split()[-1]

        if pdbId not in rDic:
            rDic[pdbId] = {}

        if prot.isFinished() and finished:
            rDic[pdbId][protLabel] = prot

        elif not prot.isFinished() and not finished:
            rDic[pdbId][protLabel] = prot

    return rDic


def defineInput(newProtTup, oldProtTup):
    """
    Defines the input of the child protocol as the output
    of the parent protocol.
    """

    inAttr = getattr(
        newProtTup[0],
        newProtTup[1]
    )

    inAttr.set(
        oldProtTup[0]
    )

    if isinstance(
        oldProtTup[1],
        list
    ):

        for i, oName in enumerate(
            oldProtTup[1]
        ):

            inAttr[i].setExtended(
                oName
            )

    else:

        inAttr.setExtended(
            oldProtTup[1]
        )


def _waitOutput(
    proj,
    prot,
    outputAttributeName,
    sleepTime=5,
    timeOut=4000
):
    """Wait until the output is being generated by the protocol."""

    def _loadProt():

        loadedProt = getProtocolFromDb(
            prot.getProject().path,
            prot.getDbPath(),
            prot.getObjId()
        )

        loadedProt.getProject().closeMapper()
        loadedProt.closeMappers()

        return loadedProt

    counter = 1
    prot2 = _loadProt()

    numberOfSleeps = (
        timeOut / sleepTime
    )

    while (
        not prot2.hasAttribute(
            outputAttributeName
        )
        and prot2.isActive()
    ):

        time.sleep(
            sleepTime
        )

        prot2 = _loadProt()

        if counter > numberOfSleeps:

            print(
                "Timeout (%s) reached waiting for "
                "%s at %s"
                % (
                    timeOut,
                    outputAttributeName,
                    prot
                )
            )

            break

        counter += 1

    proj._updateProtocol(
        prot
    )


def removeProtocols(
    proj,
    runsDic,
    pdbId
):

    pdbId = getBaseName(
        pdbId
    )

    if pdbId in runsDic:

        prots = [
            runsDic[pdbId][
                pLabel.format(pdbId)
            ]
            for pLabel in PROT_LABELS
            if pLabel.format(pdbId)
            in runsDic[pdbId]
        ]

        c = 0

        while c < 100 and len(prots) > 0:

            nextProts = []

            for prot in prots:

                try:
                    proj.deleteProtocol(
                        prot
                    )

                except Exception:
                    nextProts.append(
                        prot
                    )

            prots = nextProts
            c += 1

        del runsDic[pdbId]

    return runsDic


def getToRunProtocolLabels(
    runsDic,
    pdbId
):
    """
    Return the labels of protocols that need to be executed.
    """

    allProts = [
        pLabel.format(pdbId)
        for pLabel in PROT_LABELS
    ]

    toRun = []

    if pdbId in runsDic:

        for pLabel in allProts:

            if pLabel not in runsDic[pdbId]:
                toRun.append(
                    pLabel
                )

    else:

        toRun = allProts

    return toRun


# ============================================================================
# Workflow runner
# ============================================================================

class WorkFlowRunner:

    def __init__(
        self,
        case,
        overwrite=False,
        slurm=False,
        log_file="benchmark_status.tsv"
    ):

        self.case = case

        self.pdbId = case["id"]
        self.category = case["category"]
        self.fastaFile = case["fasta"]
        self.af3File = case["af3"]

        self.overwrite = overwrite
        self.slurm = slurm
        self.log_file = log_file

        self.failedTools = []

    def getProjectName(self):

        category = self.category.replace(
            "protein",
            ""
        )

        return (
            f"BioFoldBenchmark_"
            f"{category}_"
            f"{self.pdbId}"
        )

    def getProject(self):

        manager = Manager()

        projectName = self.getProjectName()
        projectDir = manager.getProjectPath(
            projectName
        )

        if (
            os.path.exists(projectDir)
            and not self.overwrite
        ):

            project = manager.loadProject(
                projectName
            )

        else:

            print(
                f"Creating project {projectName}"
            )

            cleanPath(
                projectDir
            )

            project = manager.createProject(
                projectName
            )

        os.chdir(
            projectDir
        )

        return project

    def getToRunLabels(self):

        finishDic = loadRunDic(
            self.proj
        )

        failDic = loadRunDic(
            self.proj,
            finished=False
        )

        removeProtocols(
            self.proj,
            failDic,
            self.pdbId
        )

        return getToRunProtocolLabels(
            finishDic,
            self.pdbId
        )

    def safeLaunch(
        self,
        prot,
        outAttr,
        gpu=False,
        cMax=2,
        timeOut=1000
    ):

        if self.slurm:

            sendToSlurm(
                prot,
                GPU=gpu
            )

        for c in range(cMax):

            try:

                self.proj.launchProtocol(
                    prot,
                    wait=True
                )

                _waitOutput(
                    self.proj,
                    prot,
                    outAttr,
                    timeOut=timeOut
                )

                prot = getProtocolFromDb(
                    self.proj.path,
                    prot.getDbPath(),
                    prot.getObjId()
                )

                if prot.isFailed():

                    msg = "Unknown error"

                    try:
                        msg = prot.getErrorMessage()

                    except Exception:
                        pass

                    raise RuntimeError(
                        f"{prot.getObjLabel()} FAILED\n"
                        f"Reason:\n{msg}"
                    )

                if prot.isAborted():

                    raise RuntimeError(
                        f"{prot.getObjLabel()} was aborted."
                    )

                if not prot.isFinished():

                    raise RuntimeError(
                        f"{prot.getObjLabel()} ended "
                        f"with status "
                        f"{prot.getStatus()}."
                    )

                if not prot.hasAttribute(
                    outAttr
                ):

                    raise RuntimeError(
                        f"{prot.getObjLabel()} finished "
                        f"but '{outAttr}' was never created."
                    )

                time.sleep(2)

                return True

            except Exception as e:

                print(
                    f"\n{'=' * 80}"
                )

                print(
                    f"Attempt {c + 1}/{cMax}"
                )

                print(e)

                print(
                    f"{'=' * 80}\n"
                )

                time.sleep(5)

                self.failedTools.append(
                    prot.getObjLabel()
                )

        return False

    # ------------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------------

    def runImportReference(
        self,
        labIdx=0
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtImportPdb,
                inputPdbData=0,
                pdbId=self.pdbId
            )

            prot.setObjLabel(
                label
            )

            self.safeLaunch(
                prot,
                "outputPdb"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    def runPrepareReference(
        self,
        protRef,
        labIdx=1
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtChemPrepareReceptor,
                usePDBFixer=False,
                repNonStd=True
            )

            prot.setObjLabel(
                label
            )

            defineInput(
                (prot, "inputAtomStruct"),
                (protRef, "outputPdb")
            )

            self.safeLaunch(
                prot,
                "outputStructure"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    # ------------------------------------------------------------------------
    # AF3
    # ------------------------------------------------------------------------

    def runImportAF3(
        self,
        labIdx=6
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtImportPredictions,
                inputOrigin=0,
                folder=self.af3File
            )

            prot.setObjLabel(
                label
            )

            self.safeLaunch(
                prot,
                "outputBestAtomStruct"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    # ------------------------------------------------------------------------
    # Predictors
    # ------------------------------------------------------------------------

    def runBoltz(
        self,
        labIdx=2
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtBoltz,
                inputOrigin=3,
                file=self.fastaFile,
                diffusionSamples=1
            )

            prot.gpuList.set(
                "1"
            )

            prot.setObjLabel(
                label
            )

            self.safeLaunch(
                prot,
                "outputBestAtomStruct"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    def runChai(
        self,
        labIdx=3
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtChai,
                inputOrigin=3,
                file=self.fastaFile,
                diffNsamples=1
            )

            prot.gpuList.set(
                "1"
            )

            prot.setObjLabel(
                label
            )

            self.safeLaunch(
                prot,
                "outputBestAtomStruct"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    def runProtenix(
        self,
        labIdx=4
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtProtenix,
                inputOrigin=3,
                file=self.fastaFile,
                model=1,
                sample=1
            )

            prot.gpuList.set(
                "1"
            )

            prot.setObjLabel(
                label
            )

            self.safeLaunch(
                prot,
                "outputBestAtomStruct"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    def runIF(
        self,
        labIdx=5
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtIntelliFold,
                inputOrigin=3,
                file=self.fastaFile,
                diffusionSamples=1
            )

            prot.gpuList.set(
                "1"
            )

            prot.setObjLabel(
                label
            )

            self.safeLaunch(
                prot,
                "outputBestAtomStruct"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    # ------------------------------------------------------------------------
    # OpenStructure
    # ------------------------------------------------------------------------

    def runOST(
        self,
        predProt,
        labelIdx
    ):

        label = PROT_LABELS[
            labelIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ProtCompareStructures
            )

            prot.setObjLabel(
                label
            )

            defineInput(
                (prot, "inputReference"),
                (
                    self.protRefPrep,
                    "outputStructure"
                )
            )

            defineInput(
                (prot, "inputModel"),
                (
                    predProt,
                    "outputBestAtomStruct"
                )
            )

            self.safeLaunch(
                prot,
                "outputAtomStruct"
            )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    # ------------------------------------------------------------------------
    # Chimera
    # ------------------------------------------------------------------------

    def runChimeraDiscrepancies(
        self,
        labIdx=12
    ):

        label = PROT_LABELS[
            labIdx
        ].format(
            self.pdbId
        )

        if label in self.toRunList:

            prot = self.proj.newProtocol(
                ChimeraProtDiscrepancies
            )

            prot.setObjLabel(
                label
            )

            defineInput(
                (prot, "reference"),
                (
                    self.protRefPrep,
                    "outputStructure"
                )
            )

            structures = [
                self.protBoltz,
                self.protChai,
                self.protProtenix,
                self.protIF,
                self.protAF3
            ]

            prot.structures.set(
                [
                    p.outputBestAtomStruct
                    for p in structures
                ]
            )

            self.proj.launchProtocol(
                prot,
                wait=True
            )

            if prot.isFailed():

                msg = "Unknown error"

                try:
                    msg = prot.getErrorMessage()

                except Exception:
                    pass

                raise RuntimeError(
                    f"{label} FAILED\n"
                    f"Reason:\n{msg}"
                )

        else:

            prot = self.runsDic[
                self.pdbId
            ][label]

        self.allProts.append(
            prot
        )

        return prot

    # ------------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------------

    def launchProtocols(self):

        self.allProts = []

        self.runsDic = loadRunDic(
            self.proj
        )

        self.toRunList = self.getToRunLabels()

        print(
            "Protocols to run:"
        )

        print(
            self.toRunList
        )

        self.protRef = (
            self.runImportReference()
        )

        self.protRefPrep = (
            self.runPrepareReference(
                self.protRef
            )
        )

        print(
            "Running AF3 import"
        )

        self.protAF3 = (
            self.runImportAF3()
        )

        print(
            "Running Boltz"
        )

        self.protBoltz = (
            self.runBoltz()
        )

        print(
            "Running Chai"
        )

        self.protChai = (
            self.runChai()
        )

        print(
            "Running Protenix"
        )

        self.protProtenix = (
            self.runProtenix()
        )

        print(
            "Running IntelliFold"
        )

        self.protIF = (
            self.runIF()
        )

        print(
            "Running OpenStructure comparison"
        )

        self.runOST(
            self.protBoltz,
            OST_LABELS["Boltz"]
        )

        self.runOST(
            self.protChai,
            OST_LABELS["Chai"]
        )

        self.runOST(
            self.protProtenix,
            OST_LABELS["Protenix"]
        )

        self.runOST(
            self.protIF,
            OST_LABELS["IntelliFold"]
        )

        self.runOST(
            self.protAF3,
            OST_LABELS["AF3"]
        )

        print(
            "Running Chimera residue-level RMSD"
        )

        self.runChimeraDiscrepancies()

    def runWorkflow(
        self,
        trials=1
    ):

        done = False
        c = 0

        while (
            not done
            and c < trials
        ):

            try:

                self.proj = (
                    self.getProject()
                )

                self.launchProtocols()

                done = True

                if self.failedTools != []:

                    fails = list(
                        dict.fromkeys(
                            x
                            for x in self.failedTools
                            if "OpenStructure"
                            not in x
                        )
                    )

                    log_case(
                        self.log_file,
                        self.pdbId,
                        self.category,
                        "FAILED",
                        str(fails)
                    )

                else:

                    log_case(
                        self.log_file,
                        self.pdbId,
                        self.category,
                        "SUCCESS"
                    )

            except Exception as e:

                log_case(
                    self.log_file,
                    self.pdbId,
                    self.category,
                    "FAILED",
                    str(e).replace(
                        "\n",
                        " "
                    )
                )

                raise

            finally:

                c += 1

        print(
            f"Finished {self.pdbId}"
        )


# ============================================================================
# Command-line interface
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the biomolecular complex structure "
            "prediction benchmark in Scipion."
        )
    )

    parser.add_argument(
        "-i",
        "--fasta-dir",
        required=True,
        help=(
            "Root directory containing the FASTA "
            "directories for each benchmark category."
        )
    )

    parser.add_argument(
        "-a",
        "--af3-dir",
        required=True,
        help=(
            "Root directory containing the AF3 "
            "prediction ZIP files."
        )
    )

    parser.add_argument(
        "-c",
        "--category",
        required=True,
        choices=EXPECTED_CATEGORIES,
        help=(
            "Benchmark category to run."
        )
    )

    parser.add_argument(
        "--ids-file",
        default=None,
        help=(
            "Optional text file containing the PDB IDs "
            "to run. If omitted, all FASTA files in "
            "the selected category are processed."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and recreate existing Scipion "
            "projects."
        )
    )

    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "Submit Scipion protocols to the configured "
            "SLURM queue."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help=(
            "Number of workflow attempts per case "
            "(default: 1)."
        )
    )

    parser.add_argument(
        "-o",
        "--log",
        default="benchmark_status.tsv",
        help=(
            "Benchmark status log file "
            "(default: benchmark_status.tsv)."
        )
    )

    args = parser.parse_args()

    fasta_dir = os.path.abspath(
        args.fasta_dir
    )

    af3_dir = os.path.abspath(
        args.af3_dir
    )

    log_file = os.path.abspath(
        args.log
    )

    if not os.path.isdir(fasta_dir):

        parser.error(
            f"FASTA directory does not exist: "
            f"{fasta_dir}"
        )

    if not os.path.isdir(af3_dir):

        parser.error(
            f"AF3 directory does not exist: "
            f"{af3_dir}"
        )

    if args.ids_file is not None:

        args.ids_file = os.path.abspath(
            args.ids_file
        )

        if not os.path.isfile(
            args.ids_file
        ):

            parser.error(
                f"ID file does not exist: "
                f"{args.ids_file}"
            )

    cases = loadCases(
        fasta_dir=fasta_dir,
        af3_dir=af3_dir,
        category=args.category,
        ids_file=args.ids_file
    )

    log_parent = os.path.dirname(
        log_file
    )

    if log_parent:

        os.makedirs(
            log_parent,
            exist_ok=True
        )

    with open(
        log_file,
        "w"
    ) as f:

        f.write(
            "PDB\tCategory\tStatus\tMessage\n"
        )

    print(
        f"{len(cases)} benchmark cases found\n"
    )

    for case in cases:

        print(
            f"----- {case['id']}"
        )

        try:

            runner = WorkFlowRunner(
                case,
                overwrite=args.overwrite,
                slurm=args.slurm,
                log_file=log_file
            )

            runner.runWorkflow(
                trials=args.trials
            )

        except Exception as e:

            print(
                f"FAILED: {case['id']}"
            )

            print(e)


if __name__ == "__main__":
    main()