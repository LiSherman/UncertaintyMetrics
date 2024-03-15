import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def cal_UCM(uncertainty,prediction,groundTruth):
    union = np.logical_or(prediction,groundTruth)
    correct = np.logical_and(np.logical_and(prediction,groundTruth),union)
    error = np.logical_and(np.logical_xor(prediction,groundTruth),union)
    cor_distr = KernelDensity(bandwidth=0.01).fit(uncertainty[np.where(correct)][:, np.newaxis])
    bins = 200
    x = np.linspace(0,1,bins)[:, np.newaxis]
    cor_distribution = np.exp(cor_distr.score_samples(x))
    err_distr = KernelDensity(bandwidth=0.01).fit(uncertainty[np.where(error)][:, np.newaxis])
    err_distribution = np.exp(err_distr.score_samples(x))
    idx = np.where((err_distribution>=cor_distribution)==True)[0]
    inser_idx = idx[0]
    UCM = (sum(err_distribution[:inser_idx])+sum(cor_distribution[inser_idx:]))/(sum(cor_distribution)+sum(err_distribution))
    threshold = inser_idx/bins
    return UCM

def cal_ESCE(uncertainty,prediction,groundTruth,threshold):
    union = np.logical_or(prediction,groundTruth)
    correct = np.logical_and(np.logical_and(prediction,groundTruth),union)
    low_uncertainty = np.logical_and(uncertainty<threshold,union)
    u_dice = np.logical_and(low_uncertainty,correct).sum()*2/(union.sum() + correct.sum())
    dice = correct.sum()*2/(union.sum()+correct.sum())
    ESCE = np.abs(u_dice-dice)
    return ESCE

def cal_HDice(uncertainty,prediction,groundTruth,threshold):
    union = np.logical_or(prediction,groundTruth)
    correct = np.logical_and(np.logical_and(prediction,groundTruth),union)
    low_uncertainty = np.logical_and(uncertainty<threshold,union)
    high_uncertainty = np.logical_and(uncertainty>threshold,union)
    reliable_correct = np.logical_and(low_uncertainty,correct)
    reliable_ratio = low_uncertainty.sum()/union.sum()
    reliable_dice = reliable_correct.sum() * 2/(low_uncertainty.sum()+np.logical_xor(correct,high_uncertainty).sum())
    Hdice = 2/(1./reliable_dice+1./reliable_ratio)
    return Hdice
