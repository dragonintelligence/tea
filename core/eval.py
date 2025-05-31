import numpy as np
import pandas as pd
import math

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from torchvision.utils import save_image
from torchmetrics.functional import calibration_error

from core.data import load_data, load_dataloader

# WRN-28-10 TIN200 Scores
err_source = {
    'gaussian_noise': {
        5: 0.9841,
        4: 0.972,
        3: 0.9509,
        2: 0.8744,
        1: 0.5859,
    },
    'shot_noise': {
        5: 0.9753,
        4: 0.9428,
        3: 0.8936,
        2: 0.7779,
        1: 0.5912,
    },
    'impulse_noise': {
        5: 0.9925,
        4: 0.9764,
        3: 0.9349,
        2: 0.7415,
        1: 0.6243,
    },
    'defocus_blur': {
        5: 0.9504,
        4: 0.9335,
        3: 0.848,
        2: 0.7469,
        1: 0.6668,
    },
    'glass_blur': {
        5: 0.9467,
        4: 0.9053,
        3: 0.8456,
        2: 0.7317,
        1: 0.6233,
    },
    'motion_blur': {
        5: 0.8419,
        4: 0.8075,
        3: 0.7598,
        2: 0.7,
        1: 0.6074,
    },
    'zoom_blur': {
        5: 0.8646,
        4: 0.8395,
        3: 0.8056,
        2: 0.773,
        1: 0.6972,
    },
    'snow': {
        5: 0.755,
        4: 0.7209,
        3: 0.6367,
        2: 0.6369,
        1: 0.5337,
    },
    'frost': {
        5: 0.6781,
        4: 0.6376,
        3: 0.6017,
        2: 0.5637,
        1: 0.5123,
    },
    'fog': {
        5: 0.7821,
        4: 0.6767,
        3: 0.576,
        2: 0.5226,
        1: 0.4827,
    },
    'brightness': {
        5: 0.7471,
        4: 0.6502,
        3: 0.5583,
        2: 0.5037,
        1: 0.4669,
    },
    'contrast': {
        5: 0.9834,
        4: 0.9231,
        3: 0.7535,
        2: 0.6435,
        1: 0.5755,
    },
    'elastic_transform': {
        5: 0.7539,
        4: 0.7008,
        3: 0.6637,
        2: 0.6018,
        1: 0.5977,
    },
    'pixelate': {
        5: 0.684,
        4: 0.6084,
        3: 0.558,
        2: 0.4932,
        1: 0.4679,
    },
    'jpeg_compression': {
        5: 0.721,
        4: 0.6096,
        3: 0.5756,
        2: 0.53,
        1: 0.527,
    },
}


def clean_accuracy(model, x, y, batch_size = 100, logger=None, device = None, ada=None, if_adapt=True, if_vis=False):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    mces=[]
    with torch.no_grad():
        energes_list=[]
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            if ada == 'source':
                output = model(x_curr)
            else:
                output = model(x_curr, if_adapt=if_adapt)

            # Handle tuple output (e.g., from UncertaintyModel)
            if isinstance(output, tuple):
                output = output[1]  # logits tensor
            acc += (output.max(1)[1] == y_curr).float().sum()
            # calculate mce
            pred = torch.softmax(output, dim=1) 
            mce = calibration_error(pred, y_curr, norm='max', task='multiclass', num_classes=output.shape[1], n_bins=10)
            mces.append(mce.detach().cpu())

    mces = np.array(mces).mean()
    return acc.item() / x.shape[0], mces

def clean_accuracy_loader(model, test_loader, logger=None, device=None, ada=None, if_adapt=True, if_vis=False):
    test_loss = 0
    correct = 0
    index = 1
    total_step = math.ceil(len(test_loader.dataset) / test_loader.batch_size)
    with torch.no_grad():
        for counter, (data, target) in enumerate(test_loader):
            logger.info("Test Batch Process: {}/{}".format(index, total_step))
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                if ada == 'source':
                    output = model(data)
                else:
                    output = model(data, if_adapt=if_adapt)
            test_loss += F.cross_entropy(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            index = index + 1
            
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc

def evaluate_ood(model, cfg, logger, device):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):
        res = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        res_mce = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        res_mean_corruption_error = 0.0
        err_model  = {}

        for c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            err_model[corruption_type] = {}

            for s, severity in enumerate(cfg.CORRUPTION.SEVERITY):
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
                x_test, y_test = load_data(cfg.CORRUPTION.DATASET+'c', cfg.CORRUPTION.NUM_EX,
                                            cfg.CORRUPTION.SEVERITY[s], cfg.DATA_DIR, False,
                                            [cfg.CORRUPTION.TYPE[c]])
                x_test, y_test = x_test.to(device), y_test.to(device)
                acc, mce = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True)           
                logger.info(f"acc % [{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]: {acc:.2%}")
                res[s, c] = acc
                logger.info(f"mce % [{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]: {mce:.2%}")
                res_mce[s, c] = mce

                error = 1.0 - acc
                err_model[corruption_type][severity] = round(error, 6)

                if cfg.MODEL.ADAPTATION == 'source':
                    if corruption_type not in err_source:
                        err_source[corruption_type] = {}
                    err_source[corruption_type][severity] = round(error, 6)


        frame = pd.DataFrame({i+1: res[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))

        frame = pd.DataFrame({i+1: res_mce[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res_mce, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))
    
        if cfg.MODEL.ADAPTATION == 'source':
            print("\n\n====== Err_source(Ck, s) ======")
            print("err_source = {")
            for corr_type, severities in err_source.items():
                print(f"    '{corr_type}': {{")
                for sev, err in severities.items():
                    print(f"        {sev}: {err},")
                print("    },")
            print("}\n==============================\n")
        else:
            # Print mean corruption error (mCE) for severity levels 1 to 5 separately
            mean_corruption_errors = []
            for sev_level in range(1, 6):
                ce_ck_sev = []
                for corruption in err_model:
                    if sev_level in err_model[corruption] and sev_level in err_source.get(corruption, {}):
                        err_m = err_model[corruption][sev_level]
                        err_s = err_source[corruption][sev_level]
                        if err_s > 0:
                            ce = (err_m / err_s) * 100
                            ce_ck_sev.append(ce)
                        else:
                            logger.info(f"Err_source({corruption}, {sev_level}) is 0, skipping.")
                    else:
                        logger.info(f"Missing severity={sev_level} for {corruption}, skipping.")
                
                acc_avg_sev = res[sev_level - 1, :].mean()
                if ce_ck_sev:
                    mce_sev = sum(ce_ck_sev) / len(ce_ck_sev)
                    mean_corruption_errors.append(mce_sev)

                    logger.info(f"Mean Corruption Error (mCE) at severity {sev_level}: {mce_sev:.2f}, Accuracy = {acc_avg_sev:.2%}")
                else:
                    logger.info(f"Unable to compute mCE at severity {sev_level}.")

            # Print mean corruption error (mCE) averaged across all severity levels (1-5)
            if mean_corruption_errors:
                mce_2_to_5 = sum(mean_corruption_errors[-4:]) / len(mean_corruption_errors[-4:])
                logger.info(f"Mean Corruption Error (mCE) averaged over severities 2-5: {mce_2_to_5:.2f}")

                mce_all = sum(mean_corruption_errors) / len(mean_corruption_errors)
                logger.info(f"Mean Corruption Error (mCE) averaged over all severities: {mce_all:.2f}")
            else:
                logger.info("No valid mean corruption errors computed.")
    elif cfg.CORRUPTION.DATASET == 'mnist':
        _, _, _, test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
        acc = clean_accuracy_loader(model, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION,  if_adapt=True, if_vis=True)
        logger.info("Test set Accuracy: {}".format(acc))
    
    elif cfg.CORRUPTION.DATASET == 'pacs':
        accs=[]
        for target_domain in cfg.CORRUPTION.TYPE:
            x_test, y_test = load_data(data = cfg.CORRUPTION.DATASET, data_dir=cfg.DATA_DIR, shuffle = True, corruptions=target_domain)
            x_test, y_test = x_test.to(device), y_test.to(device)
            out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=True)
            if cfg.MODEL.ADAPTATION == 'energy':
                acc = out[0]
            else:
                acc = out
            logger.info(f"acc % [{target_domain}]: {acc:.2%}")
            accs.append(acc)
        acc_mean=np.array(accs).mean()
        logger.info(f"mean acc:%: {acc_mean:.2%}")
    else:
        raise NotImplementedError

def evaluate_adv(base_model, model, cfg, logger, device):
        try:
            model.reset()
            logger.info("resetting model")
        except:
            logger.warning("not resetting model")

        x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
        x_test, y_test = x_test.to(device), y_test.to(device)
        adversary = AutoAttack(base_model, norm='L2', eps=0.5, version='custom', attacks_to_run=['apgd-ce'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_test, y_test)
        acc = clean_accuracy(model, x_adv, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, if_adapt=True)
        logger.info("acc:".format(acc))

def evaluate_ori(model, cfg, logger, device):
        try:
            model.reset()
            logger.info("resetting model")
        except:
            logger.warning("not resetting model")

        if 'cifar' in cfg.CORRUPTION.DATASET:
            x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
            x_test, y_test = x_test.to(device), y_test.to(device)
            out, mce = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
            if cfg.MODEL.ADAPTATION == 'energy':
                acc = out # energes is unused and throws an error
            else:
                acc = out
            logger.info("Test set Accuracy: {}".format(acc))
            # logger.info("Test set MCE: {}".format(mce))

        elif 'pacs' in cfg.CORRUPTION.DATASET:
            pass
        else:
            _,_,_,test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
            acc = clean_accuracy_loader(model, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
            logger.info("Test set Accuracy: {}".format(acc))