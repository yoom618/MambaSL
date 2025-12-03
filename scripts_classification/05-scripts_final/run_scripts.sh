UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
            "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
            "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
            "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
            "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
            "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" \
            "All_UEA30")



model="MambaSL"  # can be replaced with other DL model names (e.g. Crossformer, ...)
for dataset in ${UEA_MTSC30[@]}
do
    sh_fname="./scripts_classification/05-scripts_final/${model}/${dataset}.sh"
    out_fname="./scripts_classification/05-scripts_final/_test_results/${model}_${dataset}.out"
    echo "Running ${sh_fname}"
    echo "Result will be saved in ${out_fname}"
    nohup bash ${sh_fname} > ${out_fname}
done


############################################################################################################
UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
            "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
            "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
            "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
            "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
            "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" )

model="MambaSL"
for dataset in ${UEA_MTSC30[@]}
do
    sh_fname="./scripts_classification/05-scripts_final/${model}/${dataset}.sh"
    out_fname="./scripts_classification/05-scripts_final/_test_results/${model}_${dataset}.out"
    echo "Running ${sh_fname}"
    echo "Result will be saved in ${out_fname}"
    nohup bash ${sh_fname} > ${out_fname}
done


UEA_MTSC30=("All_UEA30")

model="MambaSL"
for dataset in ${UEA_MTSC30[@]}
do
    sh_fname="./scripts_classification/05-scripts_final/${model}/${dataset}.sh"
    out_fname="./scripts_classification/05-scripts_final/_test_results/${model}_${dataset}.out"
    echo "Running ${sh_fname}"
    echo "Result will be saved in ${out_fname}"
    nohup bash ${sh_fname} > ${out_fname}
done


############################################################################################################
UEA_MTSC30=("ADFTD" "FLAAP")
model="MambaSL"
for dataset in ${UEA_MTSC30[@]}
do
    sh_fname="./scripts_classification/05-scripts_final/${model} (additional, medformer-setting)/${dataset}.sh"
    out_fname="./scripts_classification/05-scripts_final/_test_results/${model}_${dataset}.out"
    echo "Running ${sh_fname}"
    echo "Result will be saved in ${out_fname}"
    nohup bash ${sh_fname} > ${out_fname}
done


############################################################################################################
UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
            "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
            "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
            "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
            "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
            "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" )

model="MambaSL"
for dataset in ${UEA_MTSC30[@]}
do
    for iteration in 2 3
    do
        sh_fname="./scripts_classification/05-scripts_final/${model} (additional, multilayer)/${dataset}_nlayers${iteration}.sh"
        out_fname="./scripts_classification/05-scripts_final/_test_results/${model}_${dataset}_nlayers${iteration}.out"
        echo "Running ${sh_fname}"
        echo "Result will be saved in ${out_fname}"
        nohup bash ${sh_fname} > ${out_fname}
    done
done


############################################################################################################
UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
            "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
            "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
            "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
            "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
            "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" )

model="MambaSL"
for dataset in ${UEA_MTSC30[@]}
do
    sh_fname="./scripts_classification/05-scripts_final/${model} (additional, inceptiontime-setting)/${dataset}.sh"
    out_fname="./scripts_classification/05-scripts_final/_test_results/${model}_${dataset}_trainlossonly.out"
    echo "Running ${sh_fname}"
    echo "Result will be saved in ${out_fname}"
    nohup bash ${sh_fname} > ${out_fname}
done
