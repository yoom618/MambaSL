# the best performing model for all UEA30 dataset
# if there is more than one model, we choose the one with the lowest model size or computation cost
model_name="?????"
tslib_dir="/data/yoom618/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

# ArticularyWordRecognition
dataset_name="ArticularyWordRecognition"

# AtrialFibrillation
dataset_name="AtrialFibrillation"

# BasicMotions
dataset_name="BasicMotions"

# CharacterTrajectories
dataset_name="CharacterTrajectories"

# Cricket
dataset_name="Cricket"

# DuckDuckGeese
dataset_name="DuckDuckGeese"

# EigenWorms
dataset_name="EigenWorms"

# Epilepsy
dataset_name="Epilepsy"

# ERing
dataset_name="ERing"

# EthanolConcentration
dataset_name="EthanolConcentration"

# FaceDetection
dataset_name="FaceDetection"

# FingerMovements
dataset_name="FingerMovements"

# HandMovementDirection
dataset_name="HandMovementDirection"

# Handwriting
dataset_name="Handwriting"

# Heartbeat
dataset_name="Heartbeat"

# InsectWingbeat
dataset_name="InsectWingbeat"

# JapaneseVowels
dataset_name="JapaneseVowels"

# Libras
dataset_name="Libras"

# LSST
dataset_name="LSST"

# MotorImagery
dataset_name="MotorImagery"

# NATOPS
dataset_name="NATOPS"

# PEMS-SF
dataset_name="PEMS-SF"

# PenDigits
dataset_name="PenDigits"

# PhonemeSpectra
dataset_name="PhonemeSpectra"

# RacketSports
dataset_name="RacketSports"

# SelfRegulationSCP1
dataset_name="SelfRegulationSCP1"

# SelfRegulationSCP2
dataset_name="SelfRegulationSCP2"

# SpokenArabicDigits
dataset_name="SpokenArabicDigits"

# StandWalkJump
dataset_name="StandWalkJump"

# UWaveGestureLibrary
dataset_name="UWaveGestureLibrary"