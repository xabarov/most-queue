import math
from most_queue.sim import rand_destribution as rd
from most_queue.theory import network_calc

class HyperVisor():

    def __init__(self, net_params, v1_treb):
        self.v1_treb = v1_treb
        self.net_params = net_params
        self.k_num = net_params['k_num']
        self.n_num = len(net_params['n'])
        self.b = []
        for k in range(self.k_num):
            self.b.append([])
            for m in range(self.n_num):
                params = self.net_params['serv_params'][m][k]['params']
                self.b[k].append(rd.H2_dist.calc_theory_moments(*params, 4))
        self.net_params_for_calc = [self.net_params['R'], self.b, self.net_params['n'],
                                    self.net_params['L'], self.net_params['prty'], self.net_params['nodes_prty']]

        self.num_of_arr_exceeds = 0
        self.num_of_seen_v1_prob = 0
        self.num_of_prty_success = 0
        self.last_seen_l = self.net_params['L']

    def arr_intens_meter(self, arrived, ttek, net, max_delta_to_start_prty=0.005):
        l_meter = []
        is_action = False
        for k in range(self.k_num):
            l_meter.append(arrived[k]/ttek)

        if math.fabs(max(self.last_seen_l)-max(l_meter)) > max_delta_to_start_prty:
            print("Start prty module...")
            for l in l_meter:
                print("{0:1.3f}".format(l), end='  ')
            print("\n")
            is_action = True
            self.num_of_arr_exceeds += 1
            self.monitor(l_meter, net)
        self.last_seen_l = l_meter

        return is_action

    def monitor(self, l_meter, net):

        self.net_params_for_calc[3] = l_meter
        semo_calc = network_calc.network_prty_calc(*self.net_params_for_calc)

        v1_tek = []
        k_losed = {}

        for k in range(self.k_num):
            v1_tek.append(semo_calc['v'][k][0])
            if v1_tek[k] > self.v1_treb[k]:
                k_losed[k] = v1_tek[k] - self.v1_treb[k]

        if len(k_losed) == 0:
            return None

        self.num_of_seen_v1_prob += 1

        k_losed_sorted = []
        listofTuples = sorted(k_losed.items(), key=lambda x: x[1], reverse=True)
        # Iterate over the sorted sequence
        for elem in listofTuples:
            k_losed_sorted.append([elem[0], elem[1]])

        for j_delta in k_losed_sorted:
            v1_k_nodes = {}
            for m in range(self.n_num):
                v1_k_nodes[m] = semo_calc['v_node'][m][j_delta[0]][0]

            list_nodes = sorted(v1_k_nodes.items(), key=lambda x: x[1], reverse=True)
            list_nodes_sorted = []
            for elem in list_nodes:
                list_nodes_sorted.append([elem[0], elem[1]])

            is_solve_k = False

            for i in list_nodes_sorted:
                if is_solve_k:
                    break
                while(True):
                    tek_nodes_prty = self.net_params['nodes_prty'][i[0]]
                    print("Try to change PRTY. Node: ", i[0], '\nStart PRTY: ', tek_nodes_prty)
                    if tek_nodes_prty[j_delta[0]] == 0:
                        break

                    tek_nodes_prty[j_delta[0]] -= 1
                    val = tek_nodes_prty[j_delta[0]]
                    changed_class = -1
                    for s in range(len(tek_nodes_prty)):
                        if tek_nodes_prty[s] == val and s!=j_delta[0]:
                            tek_nodes_prty[s] += 1
                            changed_class = s
                            break
                    net.smos[i[0]].swop_queue(val+1, val)

                    semo_calc = network_calc.network_prty_calc(*self.net_params_for_calc)
                    k_losed_after_impov = {}
                    v1_tek_after = []

                    is_nan_v = False
                    for k in range(self.k_num):
                        v1_tek_after.append(semo_calc['v'][k][0])
                        if not v1_tek_after[k]:
                            is_nan_v = True

                    if is_nan_v:
                        tek_nodes_prty[j_delta[0]] += 1
                        tek_nodes_prty[changed_class] -= 1
                        net.smos[i[0]].swop_queue(val, val + 1)
                        break

                    for k in range(self.k_num):
                        if v1_tek_after[k] > self.v1_treb[k]:
                            k_losed_after_impov[k] = v1_tek_after[k] - self.v1_treb[k]
                    if len(k_losed_after_impov) > len(k_losed):
                        tek_nodes_prty[j_delta[0]] += 1
                        tek_nodes_prty[changed_class] -= 1
                        net.smos[i[0]].swop_queue(val, val + 1)
                        break

                    print('End PRTY: ', tek_nodes_prty)
                    if semo_calc['v'][j_delta[0]][0] < self.v1_treb[j_delta[0]]:
                        is_solve_k = True
                        print('Success!')
                        self.num_of_prty_success += 1
                        break

