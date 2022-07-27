import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns


def read_darwin_evaluations(export_path):
    """
    Extract JSON evaluations data. Darwin stores annotation information about each image in different files with the
    image name.

    :param export_path: Darwin export path

    :return: JSON string containing all evaluations
    """
    evaluations = {}
    for filename in os.listdir(export_path):
        with open(Path(export_path) / filename, "r") as file:
            evaluations[filename] = json.load(file)

    return evaluations


def create_diagnosis_df(evals):
    """
    Create a one-hot encoded diagnosis dataframe indexed by image ID and labeller. The two discard options are collapsed
    into a single column, `discard`. Evaluations with multiple diagnoses are discarded.
    :param evals: list of evaluations obtained from `read_darwin_evaluations`
    :return:
    """
    diagnosis_dict = {}
    rename_dict = {
        'disease_Acne': 'acne',
        'disease_Actinic keratosis': 'actinic_keratosis',
        'disease_Other-disease': 'other_disease',
        'disease_Psoriasis': 'psoriasis',
        'disease_Seborrheic dermatitis': 'seborrheic_dermatitis',
        'disease_Viral warts': 'viral_warts',
        'disease_Vitiligo': 'vitiligo'
    }

    image_quality_count = 0
    for evaluation_id, evaluation in evals.items():
        labeller, image_id = evaluation_id.split('_')
        diagnosis_dict[(image_id, labeller)] = {}
        selected_diagnosis = [
            annotation['name']
            for annotation in evaluation["annotations"]
            if annotation['name']
               in ['Acne', 'Actinic keratosis', 'Other-disease', 'Psoriasis', 'Seborrheic dermatitis', 'Viral warts', 'Vitiligo']
        ]
        if len(selected_diagnosis) > 0:
            diagnosis_dict[(image_id, labeller)]['disease'] = selected_diagnosis[0]
        else:
            diagnosis_dict[(image_id, labeller)]['disease'] = np.nan
            image_quality_count += 1
    print(f'{image_quality_count} evaluations dropped as Low quality')

    diagnosis_df = pd.DataFrame.from_dict(diagnosis_dict, orient='index')
    diagnosis_df = pd.get_dummies(diagnosis_df, prefix=None)
    diagnosis_df = diagnosis_df.rename(columns=rename_dict)
    # Name the index columns
    diagnosis_df.index = diagnosis_df.index.rename(['image_id', 'labeller_id'])
    return diagnosis_df


def create_characteristics_df(evals):
    characteristics_dict = {}
    characteristic_names = ['Acral distribution',
                            'Closed comedo',
                            'Convex margins',
                            'Cyst',
                            'Dark skin',
                            'Dermatoglyph disruption',
                            'Elderly',
                            'Exposed areas',
                            'Extensor sites',
                            'Female',
                            'Hyperkeratosis',
                            'Intertriginous',
                            'Köbnerization',
                            'Leukotrichia',
                            'Light skin',
                            'Macule',
                            'Male',
                            'Nodule',
                            'Open comedo',
                            'Palmo-plantar distribution',
                            'Papule',
                            'Patch',
                            'Periorificial',
                            'Plaque',
                            'Pustule',
                            'Scale',
                            'Scar',
                            'Seborrheic region',
                            'Sun damage',
                            'Symmetrical',
                            'Telangiectasia',
                            'Thrombosed capillaries',
                            'Young']
    rename_dict = {
        'Acral distribution': 'acral_distribution',
        'Closed comedo': 'closed_comedo',
        'Convex margins': 'convex_margins',
        'Cyst': 'cyst',
        'Dark skin': 'dark_skin',
        'Dermatoglyph disruption': 'dermatoglyph_disruption',
        'Elderly': 'elderly',
        'Exposed areas': 'exposed_areas',
        'Extensor sites': 'extensor_sites',
        'Female': 'female',
        'Hyperkeratosis': 'hyperkeratosis',
        'Intertriginous': 'intertriginous',
        'Köbnerization': 'koebnerization',
        'Leukotrichia': 'leukotrichia',
        'Light skin': 'light_skin',
        'Macule': 'macule',
        'Male': 'male',
        'Nodule': 'nodule',
        'Open comedo': 'open_comedo',
        'Palmo-plantar distribution': 'palmo_plantar_distribution',
        'Papule': 'papule',
        'Patch': 'patch',
        'Periorificial': 'periorificial',
        'Plaque': 'plaque',
        'Pustule': 'pustule',
        'Scale': 'scale',
        'Scar': 'scar',
        'Seborrheic region': 'seborrheic_region',
        'Sun damage': 'sun_damage',
        'Symmetrical': 'symmetrical',
        'Telangiectasia': 'telangiectasia',
        'Thrombosed capillaries': 'thrombosed_capillaries',
        'Young': 'young',
    }

    for evaluation_id, evaluation in evals.items():
        labeller, image_id = evaluation_id.split('_')
        characteristics_dict[(image_id, labeller)] = {}
        for annotation in evaluation["annotations"]:
            if annotation['name'] in characteristic_names:
                characteristics_dict[(image_id, labeller)][rename_dict[annotation['name']]] = True

    characteristics_df = pd.DataFrame.from_dict(characteristics_dict, orient='index')
    characteristics_df = characteristics_df.fillna(False)
    # Drop empty evaluations
    characteristics_df = characteristics_df.loc[(characteristics_df.sum(axis=1) != 0)]
    # Name the index columns
    characteristics_df.index = characteristics_df.index.rename(['image_id', 'labeller_id'])
    return characteristics_df


def get_overall_diagnosis_accuracy(diagnosis_df, ground_truth_df):
    gt_diagnoses_df = diagnosis_df.astype(int).idxmax(axis=1).sort_index().reset_index().set_index('image_id')
    ground_truth_df = ground_truth_df.reset_index().rename(columns={"current_filename": "Image", "diagnosis": "Class"})
    ground_truth_df["Image"] = ground_truth_df["Image"].apply(lambda x: x.split('.')[0] +'.json')
    ground_truth_df = ground_truth_df.set_index("Image")[['Class']]
    ground_truth_df["Class"] = ground_truth_df["Class"].apply(lambda x: x.lower().replace(' ', '_'))
    gt_diagnoses_df = gt_diagnoses_df.merge(ground_truth_df, left_index=True, right_index=True)
    return accuracy_score(gt_diagnoses_df['Class'].values, gt_diagnoses_df[0].values)


def get_disease_gt_performance(diagnosis_df, gt_df):
    """
    Get per-class labeller performance with regards to the ground truth. Expects diagnosis_df and gt_df to have the same
    image_ids. Returns the F1, recall, and specificity.
    :param diagnosis_df: pandas.DataFrame of labeller evaluations
    :param gt_df: pandas.DataFrame of ground truth classes
    :return: metrics dictionary
    """
    # Turn the ground truth csv into a similar format to the diagnosis_df
    ground_truth_df = gt_df.reset_index().rename(columns={"current_filename": "Image", "diagnosis": "Class"})
    ground_truth_df["Image"] = ground_truth_df["Image"].apply(lambda x: x.split('.')[0] +'.json')
    ground_truth_df = ground_truth_df.set_index("Image")[['Class']]
    ground_truth_df["Class"] = ground_truth_df["Class"].apply(lambda x: x.lower().replace(' ', '_'))

    gt_diagnoses_df = diagnosis_df.astype(int).idxmax(axis=1).sort_index().reset_index().set_index('image_id')
    diseases = diagnosis_df.columns
    labellers = diagnosis_df.reset_index().labeller_id.unique()
    metrics_dict = {
        'accuracy': {},
        'f1': {},
        'recall': {},
        'specificity': {},
        'avg_selection': {}
    }

    for disease in diseases:
        f1s = []
        recalls = []
        specificities = []
        accuracies = []
        selections = []

        for labeller in labellers:
            labeller_diagnoses = gt_diagnoses_df[gt_diagnoses_df['labeller_id'] == labeller][0]
            labeller_diagnoses = labeller_diagnoses == disease
            gt_diagnoses = ground_truth_df.reindex(index=labeller_diagnoses.index)
            gt_diagnoses = gt_diagnoses == disease

            accuracies.append(accuracy_score(gt_diagnoses, labeller_diagnoses))
            f1s.append(f1_score(gt_diagnoses, labeller_diagnoses, pos_label=True, average='binary'))
            recalls.append(recall_score(gt_diagnoses, labeller_diagnoses, pos_label=True, average='binary'))
            specificities.append(specificity(gt_diagnoses, labeller_diagnoses))
            selections.append(len(labeller_diagnoses[labeller_diagnoses == True]))

        metrics_dict['accuracy'][
            disease] = f'\${np.round(np.mean(accuracies), decimals=2)} \pm {np.round(np.std(accuracies), decimals=2)}\$'
        metrics_dict['f1'][
            disease] = f'\${np.round(np.mean(f1s), decimals=2)} \pm {np.round(np.std(f1s), decimals=2)}\$'
        metrics_dict['recall'][
            disease] = f'\${np.round(np.mean(recalls), decimals=2)} \pm {np.round(np.std(recalls), decimals=2)}\$'
        metrics_dict['specificity'][
            disease] = f'\${np.round(np.mean(specificities), decimals=2)} \pm {np.round(np.std(specificities), decimals=2)}\$'
        metrics_dict['avg_selection'][
            disease] = f'\${np.round(np.mean(selections), decimals=2)} \pm {np.round(np.std(selections), decimals=2)}\$'

    return pd.DataFrame.from_dict(metrics_dict)


# Metrics supporting probabilistic segmentation maps
fuzzy_and = lambda x, y: np.minimum(x, y)
fuzzy_or = lambda x, y: np.maximum(x, y)
fuzzy_not = lambda x: 1 - x


def pixel_metrics_fuzzy(y_true, y_pred):
    """
    Pixel-level metrics of segmentation accuracy following fuzzy logic operators.

    :param y_true: numpy.ndarray of reference segmentation, values in [0,1]
    :param y_pred: numpy.ndarray of predicted segmentation, values in [0,1]

    :return: a dictionary encoding the metrics
    """

    np.testing.assert_equal(y_true.shape, y_pred.shape, err_msg="Expecting \
    the reference and predicted segmentations to be of the same size.")

    # Check the ranges
    np.testing.assert_equal(np.logical_and(y_true >= 0, y_true <= 1).all(), True, err_msg="Expecting \
    the reference segmentations to be in the range 0 to 1.")
    np.testing.assert_equal(np.logical_and(y_pred >= 0, y_pred <= 1).all(), True, err_msg="Expecting \
    the predicted segmentations to be in the range 0 to 1.")

    TP = fuzzy_and(y_true, y_pred).sum()
    TN = fuzzy_and(fuzzy_not(y_true), fuzzy_not(y_pred)).sum()
    union = fuzzy_or(y_true, y_pred).sum()

    metrics = dict()

    # Summary metrics
    metrics["iou"] = TP / union
    metrics["dice"] = 2 * TP / (y_true.sum() + y_pred.sum())

    # Positive class metrics
    metrics["precision"] = TP / y_pred.sum()
    metrics["recall"] = TP / y_true.sum()

    # Negative class metrics
    metrics["negative_predictive_value"] = TN / fuzzy_not(y_pred).sum()
    metrics["specificity"] = TN / fuzzy_not(y_true).sum()

    return metrics


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return tn / (tn + fp)


def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return tn / (tn + fn)


def get_count_by_source(diagnoses_df, ground_truth_df):
    ground_truth_df = ground_truth_df.reset_index().rename(columns={"current_filename": "Image", "diagnosis": "Class"})
    ground_truth_df["Image"] = ground_truth_df["Image"].apply(lambda x: x.split('.')[0] + '.json')
    ground_truth_df = ground_truth_df.set_index("Image")[['Class', 'dataset']]
    ground_truth_df["Class"] = ground_truth_df["Class"].apply(lambda x: x.lower().replace(' ', '_'))
    gt_diagnoses_df = diagnoses_df.astype(int).idxmax(axis=1).sort_index().reset_index().set_index('image_id')
    ground_truth_df = ground_truth_df.reindex(index=gt_diagnoses_df.index.unique())
    ground_truth_df = ground_truth_df.reset_index().groupby('image_id').first().reset_index()
    return ground_truth_df.groupby(['Class', 'dataset']).count()


def get_dx_inter_rater_agreement(diagnoses_df):
    gt_diagnoses_df = diagnoses_df.astype(int).idxmax(axis=1).sort_index().reset_index().set_index('image_id')
    diseases = diagnoses_df.columns
    labellers = diagnoses_df.reset_index().labeller_id.unique()
    metrics_dict = {
        'accuracy': {},
        'f1': {},
        'recall': {},
        'specificity': {},
        'cohen_kappa': {}
    }

    for disease in diseases:
        accuracies = []
        f1s = []
        recalls = []
        specificities = []
        kappas = []

        for labeller in labellers:
            for labeller2 in labellers:
                if labeller != labeller2:
                    labeller_diagnoses = gt_diagnoses_df[gt_diagnoses_df['labeller_id'] == labeller][0]
                    labeller2_diagnoses = gt_diagnoses_df[gt_diagnoses_df['labeller_id'] == labeller2][0]

                    labeller_diagnoses = labeller_diagnoses.reindex(labeller_diagnoses.index.intersection(labeller2_diagnoses.index))
                    labeller2_diagnoses = labeller2_diagnoses.reindex(labeller2_diagnoses.index.intersection(labeller_diagnoses.index))

                    labeller_diagnoses = labeller_diagnoses == disease
                    labeller2_diagnoses = labeller2_diagnoses == disease

                    accuracies.append(accuracy_score(labeller2_diagnoses, labeller_diagnoses))
                    f1s.append(f1_score(labeller2_diagnoses, labeller_diagnoses, pos_label=True, average='binary'))
                    recalls.append(recall_score(labeller2_diagnoses, labeller_diagnoses, pos_label=True, average='binary'))
                    specificities.append(specificity(labeller2_diagnoses, labeller_diagnoses))
                    kappas.append(cohen_kappa_score(labeller2_diagnoses, labeller_diagnoses))

        metrics_dict['accuracy'][disease] = f'\${np.round(np.mean(accuracies), decimals=2)} \pm {np.round(np.std(accuracies), decimals=2)}\$'
        metrics_dict['f1'][
            disease] = f'\${np.round(np.mean(f1s), decimals=2)} \pm {np.round(np.std(f1s), decimals=2)}\$'
        metrics_dict['recall'][
            disease] = f'\${np.round(np.mean(recalls), decimals=2)} \pm {np.round(np.std(recalls), decimals=2)}\$'
        metrics_dict['specificity'][
            disease] = f'\${np.round(np.mean(specificities), decimals=2)} \pm {np.round(np.std(specificities), decimals=2)}\$'
        metrics_dict['cohen_kappa'][
            disease] = f'\${np.round(np.mean(kappas), decimals=2)} \pm {np.round(np.std(kappas), decimals=2)}\$'

    return pd.DataFrame.from_dict(metrics_dict)


def get_gt_labeller_performance(diagnoses_df, ground_truth_path):
    ground_truth_df = pd.read_csv(ground_truth_path)
    ground_truth_df = ground_truth_df.reset_index().rename(columns={"current_filename": "Image", "diagnosis": "Class"})
    ground_truth_df["Image"] = ground_truth_df["Image"].apply(lambda x: x.split('.')[0] + '.json')
    ground_truth_df = ground_truth_df.set_index("Image")[['Class']]
    ground_truth_df["Class"] = ground_truth_df["Class"].apply(lambda x: x.lower().replace(' ', '_'))

    diagnoses_df = diagnoses_df.astype(int).idxmax(axis=1).sort_index().reset_index().set_index('image_id')
    diseases = diagnoses_df[0].unique()
    labellers = diagnoses_df.reset_index().labeller_id.unique()

    for labeller in labellers:
        labeller_diagnoses = diagnoses_df[diagnoses_df['labeller_id'] == labeller][0]
        gt_diagnoses = ground_truth_df.reindex(index=labeller_diagnoses.index)
        print(labeller, 'accuracy:',
              np.sum([1 for x, y in zip(labeller_diagnoses.values, gt_diagnoses.values) if x == y]) / len(gt_diagnoses))
        cm = confusion_matrix(gt_diagnoses, labeller_diagnoses, labels=diseases)
        df_cm = pd.DataFrame(cm, index=diseases, columns=diseases)
        plt.figure(figsize=(5, 5))
        plt.title(labeller + ' confusion matrix')
        sns.heatmap(df_cm, annot=True)
        plt.show()


def get_characteristics_inter_rater_agreement(characteristics_df):
    characteristics = characteristics_df.columns.tolist()
    labellers = characteristics_df.reset_index().labeller_id.unique()
    gt_characteristics_df = characteristics_df.reset_index().set_index('image_id').fillna(False)

    metrics_dict = {
        'accuracy': {},
        'f1': {},
        'recall': {},
        'specificity': {},
        'cohen_kappa': {},
        'avg_selection': {},
        'selection': {}
    }

    for characteristic in characteristics:
        accuracies = []
        f1s = []
        recalls = []
        specificities = []
        kappas = []
        selections = []

        for labeller in labellers:
            for labeller2 in labellers:
                if labeller != labeller2:
                    labeller_characteristics = gt_characteristics_df[gt_characteristics_df['labeller_id'] == labeller]
                    labeller2_characteristics = gt_characteristics_df[gt_characteristics_df['labeller_id'] == labeller2]

                    labeller_characteristics = labeller_characteristics.reindex(
                        labeller_characteristics.index.intersection(labeller2_characteristics.index))
                    labeller2_characteristics = labeller2_characteristics.reindex(
                        labeller2_characteristics.index.intersection(labeller_characteristics.index))

                    labeller_characteristics = labeller_characteristics[characteristic].values.tolist()
                    labeller2_characteristics = labeller2_characteristics[characteristic].values.tolist()

                    accuracies.append(accuracy_score(labeller2_characteristics, labeller_characteristics))
                    f1s.append(f1_score(labeller2_characteristics, labeller_characteristics, pos_label=True, average='binary'))
                    recalls.append(
                        recall_score(labeller2_characteristics, labeller_characteristics, pos_label=True, average='binary'))
                    specificities.append(specificity(labeller2_characteristics, labeller_characteristics))
                    kappas.append(cohen_kappa_score(labeller2_characteristics, labeller_characteristics))

            labeller_characteristic_df = gt_characteristics_df[(gt_characteristics_df['labeller_id'] == labeller) & (gt_characteristics_df[characteristic] == True)]
            selections.append(len(labeller_characteristic_df))

        metrics_dict['accuracy'][characteristic] = f'\${np.round(np.mean(accuracies), decimals=2)} \pm {np.round(np.std(accuracies), decimals=2)}\$'
        metrics_dict['f1'][
            characteristic] = f'\${np.round(np.mean(f1s), decimals=2)} \pm {np.round(np.std(f1s), decimals=2)}\$'
        metrics_dict['recall'][
            characteristic] = f'\${np.round(np.mean(recalls), decimals=2)} \pm {np.round(np.std(recalls), decimals=2)}\$'
        metrics_dict['specificity'][
            characteristic] = f'\${np.round(np.nanmean(specificities), decimals=2)} \pm {np.round(np.nanstd(specificities), decimals=2)}\$'
        metrics_dict['avg_selection'][
            characteristic] = f'\${np.round(np.mean(selections), decimals=2)} \pm {np.round(np.std(selections), decimals=2)}\$'
        metrics_dict['cohen_kappa'][characteristic] = f'\${np.round(np.nanmean(kappas), decimals=2)} \pm {np.round(np.nanstd(kappas), decimals=2)}\$'
        metrics_dict['selection'][characteristic] = np.sum(selections)

    return pd.DataFrame.from_dict(metrics_dict).sort_values(by=['selection'], ascending=False).drop(columns=['selection'])


def get_valid_masks(masks_path, image_ids):
    images = os.listdir(masks_path)
    image_ids_no_ext = [image.split('.')[0] for image in image_ids]
    valid_masks = []
    for image in images:
        labeller, image_id, characteristic = image.split('_')
        if image_id in image_ids_no_ext:
            valid_masks.append(image)
    return valid_masks


def get_masks_df(masks_path):
    """
    Create a dataframe linking each mask path with a labeller, an image, and a characteristic.
    :param masks_path:
    :return:
    """
    masks = os.listdir(masks_path)
    masks_dict = {'labeller_id': [], 'image_id': [], 'characteristic': [], 'mask_id': []}
    for mask in masks:
        labeller_id, image_id, characteristic = mask[:-4].split('_')
        masks_dict['labeller_id'].append(labeller_id)
        masks_dict['image_id'].append(image_id)
        masks_dict['characteristic'].append(characteristic)
        masks_dict['mask_id'].append(mask)
    return pd.DataFrame.from_dict(masks_dict)


def get_mask_paired_interrater_agreement(masks_path, valid_image_ids):
    """
    Only look at the pairs where both agreed.
    :param masks_path:
    :param masks_metadata_path:
    :param valid_image_ids:
    :return:
    """
    # Exclude discarded images
    images = get_valid_masks(masks_path, valid_image_ids)
    image_ids = set([image.split('_')[1] for image in images])
    masks_info_df = get_masks_df(masks_path)

    metrics_dict = {}

    for image in image_ids:
        # Find all characteristics and labellers
        image_info_df = masks_info_df[masks_info_df['image_id'] == image]

        for characteristic in image_info_df['characteristic'].unique():
            # Get all labellers that outlined this characteristic
            image_char_info_df = image_info_df[image_info_df['characteristic'] == characteristic]

            for labeller in image_char_info_df['labeller_id'].unique():
                for labeller_ref in image_char_info_df['labeller_id'].unique():
                    if labeller != labeller_ref:
                        try:
                            # Get each labeller's characteristic segmentation file path
                            segmentation_path = \
                                image_char_info_df[image_char_info_df['labeller_id'] == labeller].mask_id.values[0]
                            ref_segmentation_path = \
                                image_char_info_df[image_char_info_df['labeller_id'] == labeller_ref].mask_id.values[0]

                            segmentation = plt.imread(str(Path(masks_path) / segmentation_path))
                            ref_segmentation = plt.imread(str(Path(masks_path) / ref_segmentation_path))

                            # Get fuzzy metrics for the outline pair
                            pixel_metrics = pixel_metrics_fuzzy(np.array(ref_segmentation > 0),
                                                                np.array(segmentation > 0))

                            # Save all metrics for this pair in the metrics_dict
                            metrics_dict[(image, characteristic, labeller, labeller_ref)] = {}

                            metrics_dict[(image, characteristic, labeller, labeller_ref)]['f1'] = pixel_metrics['iou']
                            metrics_dict[(image, characteristic, labeller, labeller_ref)]['sensitivity'] = \
                                pixel_metrics['recall']
                            metrics_dict[(image, characteristic, labeller, labeller_ref)]['specificity'] = \
                                pixel_metrics['specificity']
                        except FileNotFoundError:
                            print(segmentation_path, ref_segmentation_path, 'not found')
    return pd.DataFrame.from_dict(metrics_dict, orient='index')


def get_mask_interrater_agreement(masks_path, valid_image_ids):
    if not os.path.isfile('./mask_paired_interrater_df.csv'):
        paired_df = get_mask_paired_interrater_agreement(masks_path, valid_image_ids)
        paired_df.to_csv('mask_paired_interrater_df.csv')
    paired_df = pd.read_csv('mask_paired_interrater_df.csv')
    paired_df = paired_df.rename(columns={
        'Unnamed: 0': 'image_id',
        'Unnamed: 1': 'characteristic',
        'Unnamed: 2': 'labeller',
        'Unnamed: 3': 'labeller_ref',
    })
    characteristics = paired_df['characteristic'].unique().tolist()

    metrics_dict = {
        'f1': {},
        'sensitivity': {},
        'specificity': {},
    }

    for characteristic in characteristics:
        sensitivity_mean = paired_df[paired_df['characteristic'] == characteristic]['sensitivity'].mean()
        sensitivity_std = paired_df[paired_df['characteristic'] == characteristic]['sensitivity'].std()
        specificity_mean = paired_df[paired_df['characteristic'] == characteristic]['specificity'].mean()
        specificity_std = paired_df[paired_df['characteristic'] == characteristic]['specificity'].std()
        f1_mean = paired_df[paired_df['characteristic'] == characteristic]['f1'].mean()
        f1_std = paired_df[paired_df['characteristic'] == characteristic]['f1'].std()

        metrics_dict['sensitivity'][
            characteristic] = f'\${np.round(sensitivity_mean, 2)} \pm {np.round(sensitivity_std, 2)}\$'
        metrics_dict['specificity'][
            characteristic] = f'\${np.round(specificity_mean, 2)} \pm {np.round(specificity_std, 2)}\$'
        metrics_dict['f1'][characteristic] = f'\${np.round(f1_mean, 2)} \pm {np.round(f1_std, 2)}\$'

    return pd.DataFrame.from_dict(metrics_dict)
