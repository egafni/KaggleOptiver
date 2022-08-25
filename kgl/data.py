from kgl.utils import timer, make_features, make_features_v2, DataBlock, DATA_DIR
import os

import pandas as pd


def get_df(train, USE_PRECOMPUTE_FEATURES=True, IS_1ST_STAGE=True, MEMORY_TEST_MODE=False):
    if USE_PRECOMPUTE_FEATURES:
        with timer('load feather'):
            df = pd.read_feather(os.path.join('features_v2.f'))
    else:
        # this loads a
        df = make_features(train, DataBlock.TRAIN)
        # v2
        df = make_features_v2(df, DataBlock.TRAIN)

    df.to_feather('features_v2.f')  # save cache

    test = pd.read_csv(os.path.join(DATA_DIR, 'optiver-realized-volatility-prediction', 'test.csv'))
    if len(test) == 3:
        print('is 1st stage')
        IS_1ST_STAGE = True

    if IS_1ST_STAGE and MEMORY_TEST_MODE:
        print('use copy of training data as test data to immitate 2nd stage RAM usage.')
        test_df = df.iloc[:170000].copy()
        test_df['time_id'] += 32767
        test_df['row_id'] = ''
    else:
        test_df = make_features(test, DataBlock.TEST)
        test_df = make_features_v2(test_df, DataBlock.TEST)

    print(df.shape)
    print(test_df.shape)
    df = pd.concat([df, test_df.drop('row_id', axis=1)]).reset_index(drop=True)
    return df
