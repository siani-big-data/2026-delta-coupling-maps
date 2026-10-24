REFERENCE_MODELS = ("clinical", "scalar", "dcm")
ABLATION_SPECS = {
    "covariates_only": ("clinical", "covariates", True, (0, 1)),
    "clinical_only": ("clinical", "clinical", False, (0, 1)),
    "scalar_only": ("scalar", "scalar", False, (0, 1)),
    "dcm_only": ("dcm", "dcm", False, (0, 1)),
    "dcm_contemporaneous_covariates": ("dcm", "dcm", True, (0,)),
    "dcm_aligned_covariates": ("dcm", "dcm", True, (1,)),
}
MODEL_INPUTS = {name: name for name in REFERENCE_MODELS}
MODEL_INPUTS.update({name: spec[0] for name, spec in ABLATION_SPECS.items()})
MODEL_INTERFACES = {
    "clinical": "ClinicalReferenceModel",
    "scalar": "ScalarReferenceModel",
    "dcm": "DCMReferenceModel",
}
MODEL_INTERFACES.update({name: "AblationReferenceModel" for name in ABLATION_SPECS})
