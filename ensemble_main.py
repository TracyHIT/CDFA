import os 
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TOKENIZERS_PARALLELISM"]= 'false'
import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold
import glob
import sys
from dataset import * 
from torch.utils.data import DataLoader
from dgllife.utils import EarlyStopping, Meter,RandomSplitter
from prettytable import PrettyTable
from tqdm import tqdm 
# from mordred import Calculator, descriptors
from rdkit import Chem
from adan import * 
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from dgllife.utils import smiles_to_bigraph,WeaveAtomFeaturizer,CanonicalBondFeaturizer,CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer,AttentiveFPBondFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer 
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error
from scipy import stats
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from map4 import * 
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors
from rdkit.Chem.rdMolDescriptors import (GetFeatureInvariants,
                                         GetConnectivityInvariants)
from cross_trm_net import * 
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Sheridan
from transformers import get_linear_schedule_with_warmup
from map4 import * 
def map4fp(mol):
    calculator = MAP4Calculator(1024, 2, False, False)
    fingerprints = calculator.calculate(mol)
    return np.array(fingerprints)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def compute_cls_metrics(y_true, y_prob):
    
    y_pred = np.array(y_prob) > 0.5
   
    roc_auc = roc_auc_score(y_true, y_prob)
   
    F1 = f1_score(y_true, y_pred, average = 'binary')
   
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    

    return   F1, roc_auc, mcc,  tn, fp, fn, tp


def compute_reg_metrics(y_true, y_prob):
    y_true = y_true.flatten().astype(float)
    y_prob = y_prob.flatten().astype(float)
     
    r2 = r2_score(y_true, y_prob)
    r, _ =stats.pearsonr(y_true, y_prob)
    rmse = mean_squared_error(y_true, y_prob, squared=False)
    mae = mean_absolute_error(y_true, y_prob)
    return  r2, r, rmse, mae


# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

nbits = 1024
fpFunc_dict = {}
fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)
fpFunc_dict['ErGF'] = lambda m:  AllChem.GetErGFingerprint(m,fuzzIncrement=0.3,maxPath=21,minPath=1)
 

def molecule_to_apdp(
    molecule,
    fingerprint_n_bits=nbits,
    verbose = False):
    # https://arxiv.org/pdf/1903.11789.pdf
 
    ap_fp = Pairs.GetAtomPairFingerprint(molecule)
    dp_fp = Sheridan.GetBPFingerprint(molecule)
    #ap_fp.GetLength() == 8388608
    #dp_fp.GetLength() == 8388608
    #16777216 = 8388608 + 8388608

    fingerprint = np.zeros(fingerprint_n_bits)
    for i in ap_fp.GetNonzeroElements().keys():
        fingerprint[i % fingerprint_n_bits] = 1
    for i in dp_fp.GetNonzeroElements().keys():
        fingerprint[(i + 8388608) % fingerprint_n_bits] = 1
    return fingerprint
 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
 


'''--------'''
def get_data(dataset):
    if dataset == 'ONEIL':
        drug_smiles_file = '../Data/ONEIL-COSMIC/drug_smiles.csv'
        cline_feature_file = '../Data/ONEIL-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../Data/ONEIL-COSMIC/drug_synergy.csv'
        cell_prot_emb_file =  '/home/wjj/data/drugsyn/ONEIL_cell_protein_hand_embedding.pkl'
    else:
        drug_smiles_file = '../Data/ALMANAC-COSMIC/drug_smiles.csv'
        cline_feature_file = '../Data/ALMANAC-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../Data/ALMANAC-COSMIC/drug_synergy.csv'
        cell_prot_emb_file =  '/home/wjj/data/drugsyn/ALMANAC_cell_protein_hand_embedding.pkl'


    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drugid2smiles = dict(zip(drug['pubchemid'], drug['isosmiles']))
    drug2fp = {}
    drug2fp2 = {}
    drug2fp3 = {}
    drug2fp4 = {}
    for smile in tqdm(drug['isosmiles'].values):
        mol = Chem.MolFromSmiles(smile)
        # drug2fp[smile] = molecule_to_apdp(mol)
        drug2fp4[smile] = np.array(fpFunc_dict['ecfp4'](mol)).flatten().astype(np.float32)
        drug2fp2[smile] = np.array(fpFunc_dict['hashtt'](mol)).flatten().astype(np.float32)
        drug2fp3[smile] = np.array(fpFunc_dict['maccs'](mol)).flatten().astype(np.float32)
        mol = Chem.MolFromSmiles(smile)
        drug2fp[smile] = map4fp(mol).flatten().astype(np.float32)
        # print(drug2descriptors[smile].shape)
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    # print('genedata',gene_data)

    # drug_num = len(drugid2smiles.keys())

    cline_num = len(gene_data.index)
    # print('gene_data.index')
    # print(gene_data.index)
    cline2id = dict(zip(gene_data.index, range(cline_num))) ##给每个细胞系编号
     
    id2cline = {val: key for key, val in cline2id.items()} ##给每个编号返回细胞系
    # print('gene_data.values')
    # print(gene_data.values)
    cline2vec = dict(zip(gene_data.index, np.array(gene_data.values, dtype='float32')))
    # print('cline2vec')
    # print(cline2vec)
    # clineid2vec = {key: cline2vec[cline] for (key, cline) in id2cline.items()}
    
    # gene_f = pd.read_csv('../gene_expr_sparse.csv', sep=',', header=0, index_col=[0])
    # print(gene_f.shape)
    # mut2vec = dict(zip(gene_f.index, np.array(gene_f.values)))
    # clineid2vec = {key: mut2vec[cline] for (key, cline) in id2cline.items()}
    
    
    with open(cell_prot_emb_file, 'rb') as f:
        cline2vec = pickle.load(f)
        
     
    clineid2vec = {key: cline2vec[cline] for (key, cline) in id2cline.items()}
     
 
    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    synergy = [[row[0], row[1], cline2id[row[2]], float(row[3])] for index, row in
               synergy_load.iterrows()]
 
     
     
    return synergy, drugid2smiles, clineid2vec, drug2fp, drug2fp2,  drug2fp3, drug2fp4

def data_split(synergy, test_size, rd_seed=0):
    synergy = np.array(synergy)
    train_data, test_data = train_test_split(synergy, test_size=test_size, random_state=rd_seed)

    return train_data, test_data
 
def process_data(synergy, drug2smiles, cline2vec, task_name='classification'):
    processed_synergy = []
    # 将编号转化为对应的smile和细胞系向量
    for row in synergy:
        processed_synergy.append([drug2smiles[row[0]], drug2smiles[row[1]],
                                  cline2vec[row[2]],   float(row[3])])

    if task_name == 'classification':
        threshold = 30
        for row in processed_synergy:
            row[3] = 1 if row[3] >= threshold else 0

    return np.array(processed_synergy, dtype=object)
 

def run_a_train_epoch(device, epoch,num_epochs, model, data_loader, loss_criterion, optimizer, scheduler):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))

    aux_crt = nn.MSELoss()
    for id,  (*x, y) in tbar:

        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        optimizer.zero_grad()

        
        output   = model(*x)
         
        main_loss =  loss_criterion(output.view(-1), y.view(-1))    
        loss =  main_loss  #+ cl_loss*5

        # losses = torch.stack( (main_loss, cl_loss), 0)
        # loss = multitaskloss_instance(losses)
         

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')
        # tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}  DSLoss={main_loss.item()  :.3f}  AUX_loss={cl_loss.item()  :.3f} ')
    

def run_an_eval_epoch(model, data_loader, task_name, criterion, epoch=1):
    model.eval()
    running_loss = AverageMeter()

    with torch.no_grad():
        preds =  torch.Tensor()
        preds1 =  torch.Tensor()
        preds2 =  torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
             
            logits   =  model(*x)
            loss = loss_criterion(logits.view(-1), y.view(-1))  
            if task_name == 'classification':
                logits = torch.sigmoid(logits)
            preds = torch.cat((preds, logits.cpu()), 0)
            # preds1 = torch.cat((preds1, logits1.cpu()), 0)
            # preds2 = torch.cat((preds2, logits2.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
            running_loss.update(loss.item(), y.size(0))
        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    val_loss =  running_loss.get_average()


    return preds, trues, val_loss



def run_ensmble_eval_epoch(model_list, data_loader, task_name, criterion, save_name):

    # make sure the same y for each ensmble model, however the test dataloader can ensure this
     
    with torch.no_grad():
        preds =  torch.Tensor()
        est_preds = []
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            est_pred = []
            for model in model_list:
                model.eval()
                logits   =  model(*x)
             
                if task_name == 'classification':
                    logits = torch.sigmoid(logits)
                est_pred.append( logits.cpu() )
            est_pred = torch.stack( est_pred, 1 ) # N * 5 * 1
            est_preds.append( est_pred )
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)
        
        est_preds = torch.cat( est_preds, 0 ) # BN * 5
        preds = torch.mean(est_preds, 1)

     
        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
        # results = {}
        # results['pred_i'] = est_preds.numpy()
        # results['true'] = trues 
        # with open(save_name, 'wb') as f:  # open a text file
        #     pickle.dump(results, f) # serialize the list

    return preds, trues
 
def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

 
 

if __name__ == '__main__':
     
    import argparse
# python ensemble_main.py -m 0 -g 1
# python ensemble_main.py -m 1  -g 1
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-m', '--mode', default= '1', type=str,
                      help='dataset name')
    parser.add_argument('-g', '--gpuid', default= '0', type=str,
                      help='GPU device')
    args = parser.parse_args()
    print(args)
    mode = int(args.mode)
    if mode == 1:
        dataset_name = 'ALMANAC-COSMIC'  # or ONEIL or # ALMANAC-COSMIC
        print(dataset_name)
        task_name = 'regression' # regression classification
        cv_mode_ls = [1,2,3]
    else:
        dataset_name = 'ONEIL'  # or ONEIL or # ALMANAC-COSMIC
        print(dataset_name)
        task_name = 'regression' # regression classification
        cv_mode_ls = [1,2,3]
    
    device = torch.device("cuda:"+args.gpuid if torch.cuda.is_available() else "cpu") 
    BATCH_SIZE = 128
    
    seed = 42
    setup_seed(seed)
    for cv_mode in cv_mode_ls:
        synergy, drugid2smiles, clineid2vec, drug2fp, drug2fp2, drug2fp3, drug2fp4  = get_data(dataset_name)


        lr =1e-4
        
        if task_name == 'classification':
            t_tables = PrettyTable(['method', 'F1', 'AUC', 'MCC', 'TP', 'TN', 'FP', 'FN'  ])
        else:
            t_tables = PrettyTable(['method',   'R2', 'R', 'RMSE', 'MAE' ])
             
        t_tables.float_format = '.3'   
         
        synergy_data, independent_test = data_split(synergy, test_size=0.1, rd_seed=seed)
       

        if cv_mode == 1:  # random split
            cv_data = synergy_data
        elif cv_mode == 2:  # leave_cline
            cv_data = np.unique(synergy_data[:, 2])
        else:  # leave_comb
            cv_data = np.unique(np.vstack([synergy_data[:, 0], synergy_data[:, 1]]), axis=1).T

       
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for test_fold, (cv_index, test_index) in enumerate(kf.split(cv_data)):
             
            if cv_mode == 1:
                synergy_cv, synergy_test = cv_data[cv_index], cv_data[test_index]
            elif cv_mode == 2:
                cline_cv, cline_test = cv_data[cv_index], cv_data[test_index]
                synergy_cv = np.array([i for i in synergy_data if i[2] in cline_cv])
                synergy_test = np.array([i for i in synergy_data if i[2] in cline_test])
            else:
                pair_cv, pair_test = cv_data[cv_index], cv_data[test_index]
                synergy_cv = np.array(
                    [j for i in pair_cv for j in synergy_data if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_test = np.array(
                    [j for i in pair_test for j in synergy_data if (i[0] == j[0]) and (i[1] == j[1])])

            synergy_cv = process_data(synergy_cv, drug2smiles=drugid2smiles,
                                    cline2vec=clineid2vec,   task_name=task_name)
            synergy_test = process_data(synergy_test, drug2smiles=drugid2smiles,
                                        cline2vec=clineid2vec,   task_name=task_name)
            
            synergy_independent_test = process_data(independent_test, drug2smiles=drugid2smiles,
                                        cline2vec=clineid2vec,   task_name=task_name)

            synergy_train, synergy_validation = data_split(synergy_cv, test_size=0.1, rd_seed=seed)


             

            trn_ds = AllTrmDatasetV4(synergy_train[:,0], synergy_train[:,1], synergy_train[:,2], synergy_train[:,3],     drug2fp, drug2fp2 , drug2fp3, drug2fp4 )
            val_ds = AllTrmDatasetV4(synergy_validation[:,0], synergy_validation[:,1], synergy_validation[:,2], synergy_validation[:,3],    drug2fp, drug2fp2, drug2fp3, drug2fp4)
            test_ds = AllTrmDatasetV4(synergy_test[:,0], synergy_test[:,1], synergy_test[:,2], synergy_test[:,3],     drug2fp, drug2fp2, drug2fp3, drug2fp4)
            independent_test_ds = AllTrmDatasetV4(synergy_independent_test[:,0], synergy_independent_test[:,1], synergy_independent_test[:,2], synergy_independent_test[:,3],     drug2fp, drug2fp2, drug2fp3, drug2fp4)
 

            train_loader = DataLoader(trn_ds, batch_size= BATCH_SIZE,   shuffle=True, num_workers=8 )
            test_loader = DataLoader(test_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=1  )
            valid_loader = DataLoader(val_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=8 )
            independent_test_loader = DataLoader(independent_test_ds, batch_size= BATCH_SIZE, shuffle=False, num_workers=1  )

            n_ensembles = 5
            test_preds, test_ys = [], []
            for n_e in range(n_ensembles):
                model = CLDTrmDDSModelFinal(device).to(device)
              
                optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-2  )
               
                stopper = EarlyStopping(mode='lower', patience=25, filename='CLDTrmDDSModelV2-' + task_name + dataset_name+str(cv_mode)+str(n_e))
                num_epochs = 1000
                if task_name == 'classification':
                    loss_criterion = nn.BCEWithLogitsLoss()
                else:
                    # loss_criterion = nn.HuberLoss(delta=5)
                    loss_criterion = nn.MSELoss()
    
 
                scheduler = None
                # print(scheduler.get_lr(),'----------')
                for epoch in range(num_epochs):
                    # Train
                    
                    run_a_train_epoch(device, epoch,num_epochs, model, train_loader, loss_criterion, optimizer, scheduler)
                    # Validation and early stop
                    val_pred, val_true, val_loss = run_an_eval_epoch(model, valid_loader,task_name, loss_criterion, epoch)
                    
                    if task_name == 'classification':
                        e_tables = PrettyTable(['epoch', 'F1', 'AUC', 'MCC', 'TP', 'TN', 'FP', 'FN'  ])
                        F1, roc_auc, mcc,  tn, fp, fn, tp = compute_cls_metrics(val_true,val_pred)
                        row = [epoch, F1, roc_auc, mcc,  tp, tn, fp, fn]
                    else:
                        e_tables = PrettyTable(['epoch-'+str(cv_mode),  'R2', 'R','RMSE', 'MAE' ])
                        r2, r, rmse, mae =  compute_reg_metrics(val_true,val_pred)
                        row = [epoch, r2, r, rmse, mae]

                    early_stop = stopper.step(val_loss, model)
                    e_tables.float_format = '.3' 
                    
                    e_tables.add_row(row)
                    print(e_tables)
                    if early_stop:
                        break
                stopper.load_checkpoint(model)
                print('---------------------------------------------------Test---------------------------------------------------')
                test_pred, test_y, test_loss= run_an_eval_epoch(model, test_loader,task_name, loss_criterion)
                
                if task_name == 'classification':
                    F1, roc_auc, mcc,  tn, fp, fn, tp = compute_cls_metrics(test_y,test_pred)
                    row = [ 'test', F1, roc_auc, mcc,  tp, tn, fp, fn]
                else:
                    r2, r, rmse, mae =  compute_reg_metrics(test_y,test_pred)
                    row = [ 'test', r2, r, rmse, mae]
                
                t_tables.add_row(row)
                

                test_pred, test_y, test_loss= run_an_eval_epoch(model, independent_test_loader,task_name, loss_criterion)
                
                if task_name == 'classification':
                    F1, roc_auc, mcc,  tn, fp, fn, tp = compute_cls_metrics(test_y,test_pred)
                    row = [ 'test', F1, roc_auc, mcc,  tp, tn, fp, fn]
                else:
                    r2, r, rmse, mae =  compute_reg_metrics(test_y,test_pred)
                    row = [ 'independent-test', r2, r, rmse, mae]
                
                t_tables.add_row(row)
                test_preds.append( test_pred )
                test_ys.append(test_y)
                print(t_tables)
                print('---------------------------------------------------Test---------------------------------------------------')
            
            esb_yp = None 
            for test_pred in test_preds:
                esb_yp =test_pred.reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, test_pred.reshape(1, -1)))
            
            test_pred = np.mean(esb_yp, axis=0)
            if task_name == 'classification':
                auc, aupr, f1_score, acc = compute_cls_metrics(test_y,test_pred)
                test_mean += np.array([auc, aupr, f1_score, acc])
                row_test = [ 'test', auc, aupr, f1_score, acc]

                
            else:
                r2, r, rmse, mae  = compute_reg_metrics(test_y,test_pred)
                row_test = [ 'mean-ensemble', r2, r, rmse, mae]
            t_tables.add_row(row_test)
            
            model_list = []
            for n_e in range(n_ensembles):
                model_path = 'CLDTrmDDSModelV2-' + task_name + dataset_name+str(cv_mode)+str(n_e)
                model = CLDTrmDDSModelFinal(device).to(device)
                model.load_state_dict(torch.load(model_path)['model_state_dict'])
                model_list.append(model)

            save_path = 'ensemble_results/bert-prot-'
            # run_ensmble_eval_epoch(model_list,valid_loader,task_name, loss_criterion, save_path  +'val-' + task_name + dataset_name+str(cv_mode)+'-test_fold'+str(test_fold))
            # # '/home/wjj/data/drugsyn/models/CLDTrmDDSModelV2-' + task_name + dataset_name+str(cv_mode)
            test_pred, test_y = run_ensmble_eval_epoch(model_list,test_loader,task_name, loss_criterion, save_path  +'test-' + task_name + dataset_name+str(cv_mode)+'-test_fold'+str(test_fold))
            r2, r, rmse, mae =  compute_reg_metrics(test_y,test_pred)
            row_test = [ 'mean-ensemble2', r2, r, rmse, mae]
            t_tables.add_row(row_test)

            test_pred, test_y = run_ensmble_eval_epoch(model_list,independent_test_loader,task_name, loss_criterion, save_path  +'test-' + task_name + dataset_name+str(cv_mode)+'-test_fold'+str(test_fold))
            r2, r, rmse, mae =  compute_reg_metrics(test_y,test_pred)
            row_test = [ 'mean-ensemble2-idtest', r2, r, rmse, mae]
            t_tables.add_row(row_test)
            print(t_tables)
        results_filename = 'result/prot-' + task_name + '-' + dataset_name+ '-' + str(cv_mode) + '.csv'
        ptable_to_csv(t_tables, results_filename)