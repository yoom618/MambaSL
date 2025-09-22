# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")  # GPU 0
# UEA_MTSC30=("DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" )  # GPU 0
# UEA_MTSC30=("FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat")  # GPU 1
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery")  # GPU 1
# UEA_MTSC30=("NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports")  # GPU 2
# UEA_MTSC30=("SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary")  # GPU 2
# UEA_MTSC30=("EigenWorms")  # GPU 3


UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
    "DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" \
    "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
    "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
    "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
    "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" \
    "EigenWorms")

exp="proposed"
model="MambaSL_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    datasetexp="${dataset}_${exp}"
    echo "Running ./scripts_classification/scripts_mamba/${exp}/${model}_${datasetexp}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${datasetexp}.out"
    nohup bash ./scripts_classification/scripts_mamba/${exp}/${model}_${datasetexp}.sh > ./scripts_classification/results/${model}_${datasetexp}.out &
done