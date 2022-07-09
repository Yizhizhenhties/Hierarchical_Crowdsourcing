import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector,RandomQuerySelector
from fact import FactSet
from worker import Worker, WorkerFactory, BaseWorkerFactor
from crowsourcing_processor import BaseCrowdSourcingProcessor, CrowdSourcingProcessor, QuerySettingOption
from functools import lru_cache
from dataloader import *

np.random.seed(5)  # 控制随机种子

@lru_cache(1)
def read_raw_data(dataname:str,method:str,deta:float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # with open("./different_algorithms/datasets/disease_to_symptom.txt", "r") as f:
    with open('./different_algorithms/datasets/' + dataname + '/' + dataname +'_Inputdata'+method+'.txt', "r") as f:
        raw_lines = f.readlines()
    idxes = []
    prior_p = []
    labels = []
    index = 0
    while index != len(raw_lines):
        idx = raw_lines[index]
        index += 1
        p = raw_lines[index].strip().split(" ")
        # p = raw_lines[index].strip().split("\t")
        p = [float(i) for i in p]
        p = np.asarray(p)
        index += 1
        l = raw_lines[index].strip().split(" ")
        # l = raw_lines[index].strip().split("\t")

        l = [int(i) for i in l]
        l = np.asarray(l)
        idxes.append(np.asarray([int(idx)]))
        prior_p.append(p)
        labels.append(l)
        index += 1
    return idxes, prior_p, labels
    f.close()


def get_query_selection_num() -> int:
    return 1


def get_budget() -> int:
    return 167 * k


def get_factset(split_idx: int) -> "FactSet":
    idxes, prior_p, labels = read_raw_data()
    p = (1 - labels[split_idx]) * np.array([1 - prior_p[split_idx], prior_p[split_idx]])
    p += (labels[split_idx]) * np.array([prior_p[split_idx], 1 - prior_p[split_idx]])
    return FactSet(
        facts=np.array([[0], [1]]),
        prior_p=p,
        ground_true=labels[0][split_idx],
    )


def get_processor(worker_factory: WorkerFactory, query_selector: QuerySelector,dataname:str,deta:float) -> CrowdSourcingProcessor:
    processor = BaseCrowdSourcingProcessor(worker_factory, query_selector,
                                           get_query_selection_num())
    processor.setDataname(dataname,deta)
    processor.budgets = get_budget()
    return processor


def start(kv:int, dataname:str, method:str,deta:float):
    acc_ = 0.9
    selector_id = 2
    k_ = 2
    expert_num = 2
    # 修改为正态分布的平均值
    def mean_dist() -> float:
        return acc_
    # print("acc_: "+str(acc_)+" selector_id: "+ str(selector_id)+" k_: "+str(k_))

    worker_factor = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=[[0, 1]])
    if selector_id == 1:
        query_selector = BaseQuerySelector()
    elif selector_id == 2:
        query_selector = GreedyQuerySelector()
    else:
        query_selector = RandomQuerySelector()


    processor = get_processor(worker_factor, query_selector,dataname,deta)
    a, b, c = read_raw_data(dataname,method,deta)

    # incre_dataloader = IncreEntropyDataloader(a, b, c)
    # normal_dataloader = NormalDataloader(a, b, c)
    # brute_dataloader = BruteChoiceDataloader(a, b, c , k=k_)
    # approx_dataloader = ApproxChoiceDataloader(a, b, c , k=k_)

    if selector_id == 1:
        s_dataloader = BruteChoiceDataloader(a, b, c, k=k_)
    elif selector_id == 2:
        s_dataloader = ApproxChoiceDataloader(a, b, c, k=k_)
    else:
        s_dataloader = NormalDataloader(a, b, c)

    time = 0
    times = []
    mean_entropies = []
    B = k_
    # for idx, prior_p, gt, mean_entropy in approx_dataloader:
    for idx, prior_p, gt, mean_entropy in s_dataloader:
        # print(idx)
        times.append(time)
        mean_entropies.append(mean_entropy)
        print("time: "+str(time)+" "+str(mean_entropy))
        prior_p = prior_p.tolist()
        gt = gt.tolist()
        if len(prior_p) == 1:
            k = 1
            if gt[0] == 1:
                factset = FactSet(np.array([[0], [1]]),
                                  prior_p=np.array([1-prior_p[0], prior_p[0]]),
                                  ground_true=1,
                                  fact_space=[[0, 1]])
                worker_factor.set_ground_true(np.array([1]))
            else:
                factset = FactSet(np.array([[0], [1]]),
                                  prior_p=np.array([1-prior_p[0], prior_p[0]]),
                                  ground_true=0,
                                  fact_space=[[0, 1]])
                worker_factor.set_ground_true(np.array([0]))
            processor.change_worker_factory(worker_factor)
            Q_op = QuerySettingOption(k)
            Q_op.set(processor)
        elif len(prior_p) == len(gt):
            k = k_
            factset_len = 2 ** len(gt)
            tmp_factset = []
            tmp_factspace = []
            s_format = '{:0' + str(len(gt)) + 'b}'
            for i in range(factset_len):
                s = s_format.format(i)
                s = list(s)
                tmp = []
                for j in range(len(s)):
                    tmp.append(int(s[j]))
                tmp_factset.append(tmp)
            for i in range(factset_len):
                tmp_factspace.append([0, 1])
            worker_factor1 = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=tmp_factspace)
            gt_str = [str(i) for i in gt]
            gt_str.reverse()
            gt_str = ''.join(gt_str)
            ground_true_int = int(gt_str, 2)
            tmp_prior_p = [1e-5 for i in range(2**len(prior_p))]
            for i in range(len(prior_p)):
                tmp_prior_p[2**i] = prior_p[i]
            factset = FactSet(np.array(tmp_factset),
                              prior_p=np.array(tmp_prior_p),
                              ground_true=ground_true_int,
                              fact_space=tmp_factspace
                              )
            # if time == 0:
            #     print(tmp_factset)
            #     test = pd.DataFrame(data=tmp_factset)
            #     test.to_csv('./different_algorithms/datasets/factset10.csv', index=False, encoding='gbk')

            gt.reverse()
            worker_factor1.set_ground_true(np.array(gt))
            processor.change_worker_factory(worker_factor1)
            Q_op = QuerySettingOption(k)
            Q_op.set(processor)
        else:
            k = k_
            factset_len = len(prior_p)
            tmp_factset = []
            tmp_factspace = []
            s_format = '{:0'+str(len(gt))+'b}'
            for i in range(factset_len):
                s = s_format.format(i)
                s = list(s)
                tmp = []
                for j in range(len(s)):
                    tmp.append(int(s[j]))
                tmp_factset.append(tmp)
            for i in range(factset_len):
                tmp_factspace.append([0, 1])
            worker_factor1 = BaseWorkerFactor(accuracy_sampler=mean_dist, fact_space=tmp_factspace)
            gt_str = [str(i) for i in gt]
            gt_str.reverse()
            gt_str = ''.join(gt_str)
            ground_true_int = int(gt_str, 2)
            # print(tmp_factset)
            factset = FactSet(np.array(tmp_factset),
                              prior_p=np.array(prior_p),
                              ground_true=ground_true_int,
                              fact_space=tmp_factspace
                              )
            gt.reverse()
            worker_factor1.set_ground_true(np.array(gt))
            processor.change_worker_factory(worker_factor1)
            Q_op = QuerySettingOption(k)
            Q_op.set(processor)
        processor.init_task(factset)
        processor.start_task(B, k, expert_num, idx)
        post_p = factset.get_prior_p()
        if len(prior_p) == 1:
            s_dataloader.add_data(idx, np.asarray([post_p.tolist()[0]]), np.asarray(gt))
            # approx_dataloader.add_data(idx, np.asarray([post_p.tolist()[0]]), np.asarray(gt))
        else:
            s_dataloader.add_data(idx, post_p, np.asarray(gt))
            # approx_dataloader.add_data(idx, post_p, np.asarray(gt))
        time += 1
        if processor.is_end:
            break
    # 写入处理后原数据的post_p(txt)
    if selector_id == 1:
        s_dataloader.wirte_post_p_to_txt(dataname + '/' + dataname + '_' + str(acc_)+'acc+brute(k='+str(k_)+')_s')
        # s_dataloader.wirte_post_p_to_txt('test')
    elif selector_id == 2:
        s_dataloader.wirte_post_p_to_txt(dataname + '/' + dataname + '_' + str(acc_)+'acc+approx(k='+str(kv)+')'+ method + '_k='+str(k_))
    else:
        s_dataloader.wirte_post_p_to_txt(dataname + '/' + dataname + '_' + str(acc_)+'acc+random(k='+str(k_)+')_s')
    # 写入budget-mean_entropy的csv
    dataframe = pd.DataFrame({'time': times, 'mean_entropy': mean_entropies})
    if selector_id == 1:
        # dataframe.to_csv("./outputcsv/test", index=False, sep=',')
        dataframe.to_csv("./outputcsv/" + dataname + "/" + dataname +"_"+str(acc_)+"acc+brute(k="+str(k_) + ")_s.csv", index=False, sep=',')
    elif selector_id == 2:
        dataframe.to_csv("./outputcsv/" + dataname + "/" + dataname +"_"+str(acc_)+"acc+approx(k="+str(kv) + ")"+method+"_k="+str(k_)+".csv", index=False, sep=',')
    else:
        dataframe.to_csv("./outputcsv/" + dataname + "/" + dataname +"_"+str(acc_)+"acc+random(k="+str(k_) + ")_s.csv", index=False, sep=',')

    processor.outputcsv(kv,method)

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    deta = 0.85
    # dataname = 'rte'
    dataname = 'd_sentiment'
    # dataname = 'ZenCrowd_all'
    # method = 'DS'
    # method = 'MV'
    # method = 'GLAD'
    # method = 'BWA'
    # method = 'PM'
    method = 'EBCC'
    # method = 'BCC'
    # method = 'ZC'
    k = 1

    # for method in ['DS','MV','GLAD','BWA','PM','EBCC','BCC','ZC']:
    print('***********************:',method,k)
    start(k, dataname, method, deta)
