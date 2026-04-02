
'''
加载example
import pickle as pk
with open("./example.pkl","rb") as f:
    result_struct = pk.load(f)
'''

class dynamic_result_struct():
    def __init__(self) -> None:
        # 图结构信息
        self.info_in_graph = {
            "local_efficiency": {"sub_result_sequence":[]},
            "global_efficiency": {"sub_result_sequence":[]},
            "avg_path_length": {"sub_result_sequence":[]},
            "modularity_q": {"sub_result_sequence":[]},
            "average_clustering": {"sub_result_sequence":[]}
        }
        # 节点结构信息
        self.info_in_nodes = {
            "nodal_efficiency": {"sub_result_sequence":[]},
            "degree": {"sub_result_sequence":[]},
            "clustering": {"sub_result_sequence":[]}
        }

        self.source_mat = None
    

    
class static_result_struct():
    def __init__(self, res_dict) -> None:
        self.degree = res_dict['degree']
        self.adjacency_mat = res_dict['adjacency_mat']
        self.local_efficiency = res_dict['local_efficiency']
        self.global_efficiency = res_dict['global_efficiency']
        self.nodal_efficiency = res_dict['nodal_efficiency']
        self.clustering = res_dict['clustering']
        self.average_clustering = res_dict['average_clustering']
        self.avg_path_length = res_dict['avg_path_length']
        self.modularity_q = res_dict['modularity_q']

        self.source_mat = None


    