#!/bin/bash

# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")  # GPU 0
# UEA_MTSC30=("DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" )  # GPU 0
# UEA_MTSC30=("FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat")  # GPU 1
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery")  # GPU 1
# UEA_MTSC30=("NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports")  # GPU 2
# UEA_MTSC30=("SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary")  # GPU 2
# UEA_MTSC30=("EigenWorms")  # GPU 3




# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
#     "DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" \
#     "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
#     "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
#     "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
#     "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" \
#     "EigenWorms")

exp="proposed"
model="Mamba3SL_CLS"
script_dir="./scripts_classification/scripts_mamba/Mamba3SL_fullgrid"
results_dir="./scripts_classification/results/Mamba3SL_fullgrid"
mkdir -p "${results_dir}"

for dataset in "${UEA_MTSC30[@]}"
do
    script="${script_dir}/${model}_${dataset}_${exp}.sh"
    result="${results_dir}/${model}_${dataset}_${exp}.out"
    if [[ ! -f "${script}" ]]; then
        echo "Missing generated script: ${script}" >&2
        exit 1
    fi
    echo "Running ${script}"
    echo "Result will be saved in ${result}"
    nohup bash "${script}" > "${result}" 2>&1 &
done