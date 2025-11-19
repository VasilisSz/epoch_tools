# If True, only non-Movement epochs will be shown
filter_movement = False

# Change or add states to score]
states = {
    '1': 'active_wake',
    '2': 'resting',
    '3': 'NREM',
    '4': 'REM',
    'g': 'grooming',
    'a': 'eating',
    'm': 'misc'
}

# Project specific settings for quality electrodes to visualize
quality_emg = {
    81217: "EMG_L", 81175: "EMG_R", 79592: "EMG_R", 79593: "EMG_L",
    81207: "EMG_R", 80625: "EMG_R", 80630: "EMG_R", 78211: "EMG_R",
    39489: "EMG_L", 80620: "EMG_R", 78227: "EMG_L", 78233: "EMG_L",
    39508: "EMG_R", 79604: "EMG_L", 81218: "EMG_R", 79602: "EMG_L",
    78244: "EMG_R", 81193: "EMG_L",
}

quality_ofc = {
    81217: 'Both', 81175: 'Both', 79592: 'Both', 79593: 'Both',
    81207: 'Both', 80625: 'Both', 80630: "OFC_L", 78211: "OFC_R",
    39489: 'OFC_L', 80620: "Both", 78227: 'Both', 78233: 'OFC_L',
    39508: 'OFC_R', 79604: 'OFC_L', 81218: 'OFC_R', 79602: 'OFC_L',
    78244: 'Both', 81193: 'OFC_L',
}

# Plotting visualization. Change the limits if needed
ofc_lims = (-0.001, 0.001)
emg_lims = (-0.0005, 0.0005)
motion_lims = (0, 20)