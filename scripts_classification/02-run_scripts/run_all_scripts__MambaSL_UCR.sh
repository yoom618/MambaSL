# UCR_128=("ACSF1" "Adiac" "AllGestureWiimoteX" "AllGestureWiimoteY" "AllGestureWiimoteZ")  # GPU 0
# UCR_128=("ArrowHead" "BME" "Beef" "BeetleFly" "BirdChicken")  # GPU 0
# UCR_128=("CBF" "Car" "Chinatown" "ChlorineConcentration" "CinCECGTorso")  # GPU 0
# UCR_128=("Coffee" "Computers" "CricketX" "CricketY" "CricketZ")  # GPU 0
# UCR_128=("Crop" "DiatomSizeReduction" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxOutlineCorrect" "DistalPhalanxTW")  # GPU 0
# UCR_128=("DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "ECG200" "ECG5000")  # GPU 0
# UCR_128=("ECGFiveDays" "EOGHorizontalSignal" "EOGVerticalSignal" "Earthquakes" "ElectricDevices")  # GPU 0
# UCR_128=("EthanolLevel" "FaceAll" "FaceFour" "FacesUCR" "FiftyWords")  # GPU 0
# UCR_128=("Fish" "FordA" "FordB" "FreezerRegularTrain" "FreezerSmallTrain" "Fungi" )  # GPU 0
# UCR_128=("GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1")  # GPU 1
# UCR_128=("GesturePebbleZ2" "GunPoint" "GunPointAgeSpan" "GunPointMaleVersusFemale" "GunPointOldVersusYoung")  # GPU 1
# UCR_128=("Ham" "HandOutlines" "Haptics" "Herring" "HouseTwenty")  # GPU 1
# UCR_128=("InlineSkate" "InsectEPGRegularTrain" "InsectEPGSmallTrain" "InsectWingbeatSound" "ItalyPowerDemand")  # GPU 1
# UCR_128=("LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat")  # GPU 1
# UCR_128=("MedicalImages" "MelbournePedestrian" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW")  # GPU 1
# UCR_128=("MixedShapesRegularTrain" "MixedShapesSmallTrain" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2")  # GPU 1
# UCR_128=("OSULeaf" "OliveOil" "PLAID" "PhalangesOutlinesCorrect" "Phoneme")  # GPU 1
# UCR_128=("PickupGestureWiimoteZ" "PigAirwayPressure" "PigArtPressure" "PigCVP" "Plane")  # GPU 1
# UCR_128=("PowerCons" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxTW")  # GPU 1
# UCR_128=("RefrigerationDevices" "Rock" "ScreenType" "SemgHandGenderCh2" "SemgHandMovementCh2")  # GPU 2
# UCR_128=("SemgHandSubjectCh2" "ShakeGestureWiimoteZ" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances")  # GPU 2
# UCR_128=("SmoothSubspace" "SonyAIBORobotSurface1" "SonyAIBORobotSurface2" "StarLightCurves" "Strawberry")  # GPU 2
# UCR_128=("SwedishLeaf" "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2")  # GPU 2
# UCR_128=("Trace" "TwoLeadECG" "TwoPatterns" "UMD" "UWaveGestureLibraryAll")  # GPU 2
# UCR_128=("UWaveGestureLibraryX" "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine")  # GPU 2
# UCR_128=("WordSynonyms" "Worms" "WormsTwoClass" "Yoga")  # GPU 2



exp="trainlossonly"
model="MambaSL_UCR"
for dataset in ${UCR_128[@]}
do
    datasetexp="${dataset}_${exp}"
    echo "Running ./scripts_classification/scripts_mamba/ucr/${model}_${datasetexp}.sh"
    echo "Result will be saved in ./scripts_classification/results/${model}_${datasetexp}-3.out"
    nohup bash ./scripts_classification/scripts_mamba/ucr/${model}_${datasetexp}.sh > ./scripts_classification/results/${model}_${datasetexp}-3.out &
done
