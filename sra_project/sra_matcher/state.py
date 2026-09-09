"""The state contract.

Every stage reads and writes session.state through these keys and nothing else.
This is the seam between agents — keep it small and explicit so a stage can be
swapped or tested in isolation.

    intake      writes  PATIENT_PROFILE
    retrieval   reads   PATIENT_PROFILE      writes  CANDIDATES, PATIENT_PROFILE_JSON
    fanout      reads   CANDIDATES, *_JSON   writes  CURRENT_CANDIDATE_JSON (per item),
                                                     CURRENT_VERDICT (per item), VERDICTS
    ranking     reads   VERDICTS             writes  SHORTLIST
"""

PATIENT_PROFILE = "patient_profile"            # dict (PatientProfile) from intake
PATIENT_PROFILE_JSON = "patient_profile_json"  # str, for {..} instruction templating
CANDIDATES = "candidates"                      # list[dict] coarse-retrieved trials
CURRENT_CANDIDATE_JSON = "current_candidate_json"  # str, one trial, set per iteration
CURRENT_VERDICT = "current_verdict"            # dict (TrialVerdict) from reasoner, per item
VERDICTS = "verdicts"                          # list[dict] all scored trials
SHORTLIST = "shortlist"                        # list[dict] eligible+uncertain, ranked
