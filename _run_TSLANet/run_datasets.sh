# run below code in the terminal to get the results of TSLANet on UEA datasets with the same hyperparameters as the paper
# (emb_dim:256 depth:3 patch:8 mlp_ratio:3.0  were the best set of hyperparameters in the paper)
#   $ nohup bash ./_run_TSLANet/run_datasets.sh 256 3 1> /dev/null 2>&1 &

UEA_MTSC30=("ArticularyWordRecognition" "AtrialFibrillation" "BasicMotions" "CharacterTrajectories" "Cricket" \
            "DuckDuckGeese" "EigenWorms" "Epilepsy" "ERing" "EthanolConcentration" \
            "FaceDetection" "FingerMovements" "HandMovementDirection" "Handwriting" "Heartbeat" \
            "InsectWingbeat" "JapaneseVowels" "Libras" "LSST" "MotorImagery" \
            "NATOPS" "PEMS-SF" "PenDigits" "PhonemeSpectra" "RacketSports" \
            "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "StandWalkJump" "UWaveGestureLibrary" \
            )
dmodel=$1
depth=$2
mkdir -p _run_TSLANet/dm${dmodel}_depth${depth}/logs
mkdir -p _run_TSLANet/dm${dmodel}_depth${depth}/results

for dataset in ${UEA_MTSC30[@]}
do
python -u _run_TSLANet/TSLANet_classification.py \
    --data_path "/data/user/MambaSL/dataset/${dataset}" \
    --data_type uea \
    --data_name ${dataset} \
    --save_path "_run_TSLANet/dm${dmodel}_depth${depth}/logs" \
    --emb_dim ${dmodel} \
    --depth ${depth} \
    --model_id UEA_datasets \
    --gpu 3 \
    --seed 2021 > _run_TSLANet/dm${dmodel}_depth${depth}_patch8_mlp3/results/${dataset}.log 2>&1
done


######################################
###### Original run_datasets.sh ######

# for dataPath in Adiac ArrowHead Beef BeetleFly BirdChicken CBF Car ChlorineConcentration CinC_ECG_torso Coffee Computers Cricket_X Cricket_Y Cricket_Z DiatomSizeReduction DistalPhalanxOutlineAgeGroup DistalPhalanxOutlineCorrect DistalPhalanxTW Earthquakes ECG200 ECG5000 ECGFiveDays ElectricDevices FaceAll FaceFour FacesUCR FISH FordA FordB Gun_Point Ham HandOutlines Haptics Herring InlineSkate InsectWingbeatSound ItalyPowerDemand LargeKitchenAppliances Lighting2 Lighting7 MALLAT Meat MedicalImages MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW MoteStrain NonInvasiveFatalECG_Thorax1 NonInvasiveFatalECG_Thorax2 OliveOil OSULeaf PhalangesOutlinesCorrect Phoneme Plane ProximalPhalanxOutlineAgeGroup ProximalPhalanxOutlineCorrect ProximalPhalanxTW RefrigerationDevices ScreenType ShapeletSim ShapesAll SmallKitchenAppliances SonyAIBORobotSurface SonyAIBORobotSurfaceII StarLightCurves Strawberry SwedishLeaf Symbols Synthetic_control ToeSegmentation1 ToeSegmentation2 Trace TwoLeadECG Two_Patterns uWaveGestureLibrary_X uWaveGestureLibrary_Y uWaveGestureLibrary_Z uWaveGestureLibraryAll wafer Wine WordsSynonyms Worms WormsTwoClass yoga
# do
#   python -u TSLANet_classification.py \
#   --data_path set/the/path/here \
#   --emb_dim 128 \
#   --depth 2 \
#   --model_id UCR_datasets \
#   --load_from_pretrained True
# done


# for dataPath in ArticularyWordRecognition  AtrialFibrillation  BasicMotions  Cricket  Epilepsy  EthanolConcentration  FaceDetection  FingerMovements  HandMovementDirection  Handwriting  Heartbeat  InsectWingbeat  JapaneseVowels  Libras  LSST  MotorImagery  NATOPS  PEMS-SF  PenDigits  PhonemeSpectra  RacketSports  SelfRegulationSCP1  SelfRegulationSCP2  SpokenArabicDigits  StandWalkJump  UWaveGestureLibrary
# do
#   python -u TSLANet_classification.py \
#   --data_path set/the/path/here \
#   --emb_dim 256 \
#   --depth 3 \
#   --model_id UEA_datasets \

# done


# for dataPath in ucihar hhar wisdm ecg eeg
# do
#   python -u TSLANet_classification.py \
#   --data_path set/the/path/here \
#   --emb_dim 256 \
#   --depth 2 \
#   --model_id other_datasets
# done