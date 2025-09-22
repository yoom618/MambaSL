# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=("Epilepsy" "ERing" "EthanolConcentration" "FaceDetection")
# UEA_MTSC30=("FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat")
# UEA_MTSC30=("InsectWingbeat" "JapaneseVowels" "Libras" "LSST")
# UEA_MTSC30=("NATOPS" "PenDigits" "PhonemeSpectra" "RacketSports" "SelfRegulationSCP1")
# UEA_MTSC30=("SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary")
# UEA_MTSC30=("DuckDuckGeese" "EigenWorms" "MotorImagery" "PEMS-SF")

UEA_MTSC30=()
model="Crossformer_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done