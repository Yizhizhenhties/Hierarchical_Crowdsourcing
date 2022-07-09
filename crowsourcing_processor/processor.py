import logging
from typing import Any
import pandas as pd
import numpy as np

from worker import Worker, WorkerFactory
from query import QuerySelector
from fact import FactSet
import abc


class CrowdSourcingProcessor(metaclass=abc.ABCMeta):
    def __init__(self):
        self.budgets: int = -1
        self.query_selector: QuerySelector = None
        self.factset: FactSet = None
        self.is_end = False

    @abc.abstractmethod
    def init_task(self, factset: FactSet):
        pass

    @abc.abstractmethod
    def start_task(self, ):
        pass


class BaseCrowdSourcingProcessor(CrowdSourcingProcessor):
    def __init__(self,
                 worker_factory: "WorkerFactory" = None,
                 query_selector: "QuerySelector" = None,
                 query_selection_num: int = -1):
        super().__init__()
        self.worker_factory = worker_factory
        self.query_selector = query_selector
        self.query_per_iteration = query_selection_num
        self.is_end = False  # is_end 修改：用于判断budgets用完
        #### 加俩个固定的工人
        self.acc_dic = {}
        # self.ans_dic = {}
        self.w1_label = {}
        self.w2_label = {}
        self.dataname = None
        self.deta = 0.0
        ####
        self.idx = []
        self.query_idx = []

    def setDataname(self,dataname1,deta1):
        self.dataname = dataname1
        self.deta = deta1
        # with open("./different_algorithms/datasets/worker_arr_test_10fact.txt", "r") as f:
        with open("./different_algorithms/datasets/" + self.dataname + "/" + self.dataname + "_worker_acc"+str(self.deta)+".txt","r") as f:
            raw_lines = f.readlines()
            # print(raw_lines)
            i = 0

            while i != len(raw_lines):
                idx = raw_lines[i].replace('\n', '')
                i += 1
                p = raw_lines[i].strip().split(" ")
                self.acc_dic[idx] = p
                i += 1
                self.w1_label[idx] = raw_lines[i].strip().split(" ")
                # print(self.w1_label[idx])
                i += 1
                self.w2_label[idx] = raw_lines[i].strip().split(" ")
                i += 1

    def init_task(self, factset: FactSet):
        super().init_task(factset)
        assert self.worker_factory is not None
        assert self.query_selector is not None
        assert self.query_per_iteration > 0
        self.factset = factset

    def start_task(self, B, k, expert_num, id):
        # B : 每个task的预算
        # k : 选取了k个fact
        import time

        assert k != -1
        assert B > 0
        if self.budgets > 0:
            ans_p_post_o_list = [] # list of p(ans|o), such as [p(ans1|o) p(ans2|o)] 
            o_prior_p = self.factset.get_prior_p()  # p(o)
            worker_list = []
            acc_list = []
            # print(str(id[0]))
            worker_accuracy = self.acc_dic[str(id[0])]
            worker_accuracy = [float(i) for i in worker_accuracy]
            acc_index = 0
            for i in range(expert_num):
                worker = self.worker_factory.get_worker(worker_accuracy[acc_index])
                accuracy = worker.get_accuracy()
                worker_list.append(worker)
                # print(worker_list)
                acc_list.append(accuracy)
                acc_index += 1
                acc_index %= expert_num
            acc_list = np.array(acc_list)
            # print(id)
            # print(acc_list)
            query_idxes, sub_factset, ans_set_entropy = \
                    self.query_selector.select(self.factset,
                                               self.query_per_iteration,
                                               acc_list)
            # print('id:'+ str(id) + 'query_idxes:' + str(query_idxes))
            for q_id in query_idxes:
                self.idx.append(int(id))
                self.query_idx.append(q_id)

            expert_index = 0 
            if k > B:
                k = B
                self.query_per_iteration = B
            while B > 0:
                if k > B:
                    k = B
                    self.query_per_iteration = B
                # ans = worker_list[expert_index].get_answer(query_idxes)
                # print('ans:'+str(ans))
                if expert_index == 0:
                    list1 = []
                    for idx in query_idxes:
                        list1.append(int(self.w1_label[str(id[0])][idx]))
                        # list1.append(int(self.w1_label[str(id[0])][0]))
                    # print(list1)
                    ans = np.array(list1)
                if expert_index == 1:
                    list1 = []
                    for idx in query_idxes:
                        list1.append(int(self.w1_label[str(id[0])][idx]))
                    # print(list1)
                    ans = np.array(list1)
                #ans 分别是两个专家做出的回答
                ans_p, ans_p_post_o = self.factset.compute_ans_p(  # p(ans), p(ans|o)
                    ans, query_idxes, acc_list)
                ans_p_post_o_list.append(ans_p_post_o)
                self.budgets -= k
                expert_index += 1
                expert_index %= expert_num
                B -= k
            Cumprod = np.ones_like(ans_p_post_o_list[0])
            for i in ans_p_post_o_list:
                Cumprod = Cumprod * i
            o_p_post_ans = o_prior_p * Cumprod / (o_prior_p * Cumprod).sum()
            # print(o_p_post_ans)
            self.factset.set_prior_p(o_p_post_ans)
            #
        else:
            self.is_end = True

    def outputcsv(self,k,method):
        dataframe = pd.DataFrame({'id': self.idx, 'query_idx': self.query_idx})
        dataframe.to_csv("./different_algorithms/datasets/" + self.dataname + "/" + self.dataname +"_exworker_select(k="+str(k)+")"+method+"_"+str(self.deta)+".csv", index=False, sep=',')

    @staticmethod
    def from_options(*options: "ProcessorOption"
                     ) -> "BaseCrowdSourcingProcessor":
        p = BaseCrowdSourcingProcessor()
        for o in options:
            p = o.set(p)
        return p

    def change_worker_factory(self, new_worker_factory):
        self.worker_factory = new_worker_factory


class ProcessorOption(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set(self, processor: CrowdSourcingProcessor) -> "CrowdSourcingProcessor":
        pass


class QuerySettingOption(ProcessorOption):

    def __init__(self, query_selection_num: int):
        assert query_selection_num > 0
        self._query_selection_num = query_selection_num

    def set(self, processor: BaseCrowdSourcingProcessor) -> "CrowdSourcingProcessor":
        processor.query_per_iteration = self._query_selection_num
        return processor


class ProcessorResult(metaclass=abc.ABCMeta):
    """
    表示CrowSourcingProcessor的任务运行结果
    """

    @abc.abstractmethod
    def get_all(self) -> dict:
        pass

    @abc.abstractmethod
    def get(self, key) -> Any:
        pass
