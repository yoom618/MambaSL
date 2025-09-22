# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")  # GPU 0
# UEA_MTSC30=("DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" "EigenWorms")  # GPU 0
# UEA_MTSC30=("FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" )  # GPU 1
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST" )  # GPU 1
# UEA_MTSC30=("NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports")  # GPU 2
# UEA_MTSC30=("SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary")  # GPU 2
# UEA_MTSC30=("Heartbeat" "MotorImagery")  # GPU 3


# UEA_MTSC30=( "DuckDuckGeese" )  # GPU 0
# UEA_MTSC30=( "FaceDetection" )  # GPU 1
# UEA_MTSC30=("InsectWingbeat")  # GPU 1
# UEA_MTSC30=("PEMS-SF" )  # GPU 2




model="PerimidFormer_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done