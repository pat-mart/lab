import random
import unittest

from src.graph import Graph

class MyTestCase(unittest.TestCase):

    def test_star(self):
        self.graph = Graph(adjacency_dict={1: [2, 3, 5], 2: [6, 5, 3]})
        self.assertEqual(self.graph.star(self.graph.g, 1), {1: [2,3,5]})

    def test_to_list(self):
        self.test_dict: dict[int, list[int]] = {4: [3,2,0,1]}
        self.list = Graph.dict_to_list(self.test_dict)

        self.assertEqual([4,3,2,0,1], self.list)
        
    def test_dis_correct(self):
        self.mn_graph = Graph(file_path="include/road-minnesota/road-minnesota.mtx")
        td_bags, td_adj, root = self.mn_graph.dp_tree_decomp()
        pos, dis, anc = self.mn_graph.h_two_h(td_bags, td_adj, root)
        
        random_bag = random.randint(1,2500)
        
        dis_bfs = []
        
        for vertex in anc[random_bag]:
            d = Graph.bfs(self.mn_graph.g, root, vertex)[1][vertex]
            dis_bfs.append(d)
            
        dis_bfs.sort()
        
        self.assertEqual(sorted(dis[random_bag]), dis_bfs)
        

if __name__ == '__main__':
    unittest.main()
