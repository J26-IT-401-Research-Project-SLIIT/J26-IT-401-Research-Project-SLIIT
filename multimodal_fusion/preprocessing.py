"""Score-level preprocessing shared by training and serving.

Scores are already probabilities. Validation and column ordering are intentional
identity operations; no fitted normalization/imputation is needed for the tree.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .settings import FEATURES,LABELS

class ScorePreprocessor(TransformerMixin,BaseEstimator):
    def __init__(self,features=None):self.features=features

    def _validate(self,x):
        columns=list(self.features or FEATURES)
        if not isinstance(x,pd.DataFrame):x=pd.DataFrame(x,columns=columns)
        if set(x.columns)!=set(columns):raise ValueError('Score schema mismatch: provide exactly the expected features')
        x=x.loc[:,columns].copy()
        if any(x[col].map(lambda value:isinstance(value,(bool,np.bool_))).any() for col in columns):
            raise ValueError('Boolean values are not component scores')
        try:x=x.apply(pd.to_numeric,errors='raise').astype(float)
        except (ValueError,TypeError) as error:raise ValueError('Component scores must be numeric') from error
        values=x.to_numpy()
        if not np.isfinite(values).all():raise ValueError('Missing or nonfinite component score')
        if np.any((values<0)|(values>1)):raise ValueError('Component scores must be in [0,1]; invalid values are not silently clipped')
        return x

    def fit(self,x,y=None):
        clean=self._validate(x)
        self.feature_names_in_=np.array(clean.columns,dtype=object)
        self.n_features_in_=len(clean.columns)
        self.fit_rows_=len(clean)
        self.training_min_=clean.min().to_dict();self.training_max_=clean.max().to_dict()
        return self

    def transform(self,x):
        check_is_fitted(self,'feature_names_in_')
        return self._validate(x)

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'feature_names_in_')
        return self.feature_names_in_.copy()

def inspect_dataset(frame):
    required={'synthetic_id','label','provenance',*FEATURES}
    if required-set(frame.columns):raise ValueError('Dataset is missing required columns')
    if frame.synthetic_id.isna().any() or not frame.synthetic_id.is_unique:raise ValueError('Missing or duplicate participant IDs')
    if frame.label.isna().any() or set(frame.label)-set(LABELS):raise ValueError('Unknown or missing class labels')
    # This is a data-integrity check, not fitting a transform on the whole cohort.
    cleaned=ScorePreprocessor()._validate(frame[FEATURES])
    duplicate_scores=int(cleaned.duplicated().sum())
    if duplicate_scores:raise ValueError('Duplicate score vectors require a grouping decision before splitting')
    return {'rows':len(frame),'missing_feature_cells':0,'nonfinite_or_out_of_range_cells':0,
        'duplicate_participant_ids':0,'duplicate_score_vectors':duplicate_scores,
        'features_used':FEATURES,'metadata_excluded':[c for c in frame.columns if c not in FEATURES],
        'learned_preprocessing':'None; schema/range validation and canonical column ordering',
        'scaling_for_tree':'Not applied; all inputs already have probability scale [0,1]',
        'missing_input_policy':'Return incomplete; do not silently impute',
        'split_policy':'Stratified participant split before fitting; entire sklearn pipeline cloned inside each CV fold'}
