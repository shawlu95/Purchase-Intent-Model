import matplotlib.pyplot as plt
# import scikitplot as skplt
import numpy as np

def Baseline(df_labl):
    return 100* max((np.sum(df_labl.values) / len(df_labl)),
                    (1 - (np.sum(df_labl.values) / len(df_labl))))

def log(s, filrname = "log", verbose = True):
    if verbose:
        print(s)
    f= open("%s.txt"%(filrname),"a+")
    f.write("%s \n" % (s))
    f.close() 

def plot_roc_curve(fpr, tpr, label = None, size = (12, 4), save_path = None):
    plt.figure(figsize=size)
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Receiver Operating Characteristic")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if save_path != None:
        plt.savefig(save_path)

def Smooth_Curve(points, factor = 0.9):
    smooth_points = []
    for point in points:
        if len(smooth_points) > 0:
            previous = smooth_points[-1]
            smooth_points.append(previous * factor + point * (1 - factor))
        else:
            smooth_points.append(point)
    return smooth_points

def err_analyze(target, probas, threshold, verbose = True):
    summary = {}

    baseline = max((np.sum(target) / len(target)), (1 - (np.sum(target) / len(target))))
    summary["baseline"] = baseline
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    c = 0
    y_scores = probas >= threshold
    for i in target:
        if i == 1 and y_scores[c] == 1:
            tp = tp + 1
        elif i != 0 and y_scores[c] == 0:
            fn = fn + 1
        elif i != 1 and y_scores[c] == 1:
            fp = fp + 1
        elif i == 0 and y_scores[c] == 0:
            tn = tn + 1
        c = c + 1

    summary["tp"] = tp
    summary["tn"] = tn
    summary["fp"] = fp
    summary["fn"] = fn
    
    tpr = None
    if tp + fn != 0:
        tpr = float(tp) / (tp + fn)
    summary["tpr"] = tpr

    fpr = None
    if fp + tn != 0: 
        fpr = float(fp) / (fp + tn)
    summary["fpr"] = fpr

    precision = None
    if tp + fp != 0:
        precision = float(tp) / (tp + fp)
    summary["precision"] = precision

    recall = None
    if tp + fn != 0:
        recall = float(tp) / (tp + fn)
    summary["recall"] = recall

    accuracy = float(tp + tn) / len(target)
    summary["acc"] = accuracy

    if verbose:
        log("baseline = %.5f"%(baseline), verbose = verbose)
        log("tp = %i"%(tp), verbose = verbose)
        log("tn = %i"%(tn), verbose = verbose)
        log("fp = %i"%(fp), verbose = verbose)
        log("fn = %i"%(fn), verbose = verbose)

        if tp + fp == 0:
            log("precision = nan", verbose = verbose)
        else:
            log("precision = %.5f"%(precision), verbose = verbose)

        if tp + fn == 0:
            log("recall = nan", verbose = verbose)
        else:
            log("recall = %.5f"%(recall), verbose = verbose)

        log("accuracy = %.5f"%(accuracy), verbose = verbose)
        log("TPR = %.5f"%(tpr), verbose = verbose)
        log("FPR = %.5f"%(fpr), verbose = verbose)
    return summary

def find_tpr_for_fpr(target_, probas_, target_fpr, verbose_ = False):
    upper_t = 1.0
    lower_t = 0.0
    mid_t = (upper_t + lower_t) / 2
    err = 1e-4
    
    s_l = err_analyze(target_, probas_, upper_t, False)
    fpr_l = s_l["fpr"]
    
    s_u = err_analyze(target_, probas_, lower_t, False)
    fpr_u = s_u["fpr"]
    
    tpr = 0
    counter = 0
    while np.abs(upper_t - lower_t) > err:
        mid_t = (upper_t + lower_t) / 2
        s_m = err_analyze(target_, probas_, mid_t, False)
        tpr_m = s_m["tpr"]
        fpr_m = s_m["fpr"]
        if fpr_m < target_fpr:
            upper_t = mid_t
        else:
            lower_t = mid_t
            
        s_l = err_analyze(target_, probas_, upper_t, False)
        tpr_l, fpr_l = s_l["tpr"], s_l["fpr"]
        
        s_u = err_analyze(target_, probas_, lower_t, False)
        tpr_u, fpr_u = s_u["tpr"], s_u["fpr"]
        
        if verbose_:
            log("fpr = (%.10f, %.10f), tpr = (%.10f, %.10f), t = (%.10f, %.10f)"%(fpr_l, fpr_u, tpr_l,
                                                                                          tpr_u, lower_t, upper_t))
        
        counter = counter + 1
        if counter == 100:
            log("Limit reached.")
            break
    if verbose_:
        log("Converged at iteration %i"%(counter))
    return (fpr_l + fpr_u)/2, (tpr_u + tpr_l)/2, (lower_t + upper_t)/2

def evaluate_model(target, probas, verbose = False):
    target_fprs = [0.001, 0.005, 0.01]
    tprs = []
    fprs = []
    ts = []
    for target_fpr in target_fprs:
        fpr, tpr, t = find_tpr_for_fpr(target, probas, verbose_ = verbose, target_fpr = target_fpr)
        tprs.append(tpr)
        fprs.append(fpr)
        ts.append(t)
        log("Target FPR = %.5f, est FPR = %.10f, est TPR = %.10f at t = %.10f"%(target_fpr, fpr, tpr, t))

    score = tprs[0] * 0.4 + (tprs[1] + tprs[2]) * 0.3
    stat = [fprs, tprs, t]
    return score, stat

def discretize(target_, probas_, verbose_ = False, n = 10):
    step = 1.0 / n
    t = step
    tprs = []
    fprs = []
    ts = []
    while t < 1:
        s = err_analyze(target_, probas_, t, False)
        tpr, fpr = s["tpr"], s["fpr"]
        if verbose_:
            log("For t = %.2f, FPR = %.5f, TPR = %.5f"%(t, fpr, tpr))
        tprs.append(tpr)
        fprs.append(fpr)
        ts.append(t)
        t = t + step
    return tprs, fprs, ts
