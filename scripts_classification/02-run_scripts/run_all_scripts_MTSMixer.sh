# UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket")
# UEA_MTSC30=("DuckDuckGeese" "Epilepsy" "ERing" "EthanolConcentration" "FaceDetection")
# UEA_MTSC30=("FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" "InsectWingbeat")
# UEA_MTSC30=("JapaneseVowels" "Libras" "LSST" "MotorImagery" "NATOPS")
# UEA_MTSC30=("PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" "UWaveGestureLibrary")
# UEA_MTSC30=("SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump")
# UEA_MTSC30=("EigenWorms")

model="MTSMixer_CLS"
for dataset in ${UEA_MTSC30[@]}
do
    echo "Running ./scripts_classification/scripts_baseline/${model}_${dataset}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${dataset}.out"
    nohup bash ./scripts_classification/scripts_baseline/${model}_${dataset}.sh > ./scripts_classification/results/${model}_${dataset}.out &
done

