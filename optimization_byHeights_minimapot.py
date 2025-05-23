#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
from deap import base, creator, tools, algorithms
import random
import os


# --- constant definition ---
um = 1e-6
e = 1.60217662e-19
Omega = 2.0 * np.pi * 22.37e6
m = 40.0 * 1.66e-27
pseud_A = e / (4.0 * m * Omega**2)
Vrf = 185
squaresize = 10
squaresize_b = 50
a = squaresize / 2
b = squaresize_b / 2

# ===================================================
# :one:create electrode
# ===================================================
def create_rectangle(x1, z1, x2, z2):
    return np.array([[z1, x1], [z2, x1], [z2, x2], [z1, x2]])

def generate_rf_electrodes(pq_starts, n_add_each, pqb_starts, nb_add_each):
    base_rf = [
        create_rectangle(100, -5500, 400, -1000),
        create_rectangle(1000, -400, 5500, -100),
        create_rectangle(100, 1000, 400, 5500),
        create_rectangle(1000, 100, 5500, 400),
        create_rectangle(-400, -5500, -100, -1000),
        create_rectangle(-5500, -400, -1000, -100),
        create_rectangle(-400, 1000, -100, 5500),
        create_rectangle(-5500, 100, -1000, 400)
    ]

    added_rf = []
    for (p0, q0), n_add in zip(pq_starts, n_add_each):
        pq_list = [(p0, q0)]
        for _ in range(n_add - 1):
            prev_p, prev_q = pq_list[-1]
            next_q = prev_q + 2 * a
            if next_q + a < 500:
                pq_list.append((prev_p, next_q))

        for p, q in pq_list:
            x1, z1 = p - a, q - a
            x2, z2 = p + a, q + a
            added_rf.extend([
                create_rectangle(x1, z1, x2, z2),
                create_rectangle(-x2, z1, -x1, z2),
                create_rectangle(x1, -z2, x2, -z1),
                create_rectangle(-x2, -z2, -x1, -z1)
            ])
            if p != q:
                added_rf.extend([
                    create_rectangle(q - a, p - a, q + a, p + a),
                    create_rectangle(-q - a, p - a, -q + a, p + a),
                    create_rectangle(q - a, -p - a, q + a, -p + a),
                    create_rectangle(-q - a, -p - a, -q + a, -p + a)
                ])

    added_rf_base = []
    for (pb0, qb0), nb_add in zip(pqb_starts, nb_add_each):
        pqb_list = [(pb0, qb0)]
        for _ in range(nb_add - 1):
            prev_pb, _ = pqb_list[-1]
            next_pb = prev_pb - 2 * b
            if abs(next_pb) + b < 5500:
                pqb_list.append((next_pb, qb0))
        for pb, qb in pqb_list:
            xb1, zb1, xb2, zb2 = pb - b, qb - b, pb + b, qb + b
            for dx, dz in [(1,1),(-1,1),(1,-1),(-1,-1)]:
                added_rf_base.append(create_rectangle(dx*xb1, dz*zb1, dx*xb2, dz*zb2))
            if pb != qb:
                for dx, dz in [(1,1),(-1,1),(1,-1),(-1,-1)]:
                    added_rf_base.append(create_rectangle(dx*(qb - b), dz*(pb - b), dx*(qb + b), dz*(pb + b)))

    return base_rf, added_rf, added_rf_base

# ===================================================
# :two: potential funtction
#  ===================================================
def extract_bounds(poly_list):
    return [[min([p[1] for p in poly]), min([p[0] for p in poly]),
             max([p[1] for p in poly]), max([p[0] for p in poly])]
            for poly in poly_list]

def pot(x, y, z, electrode, V):
    x1, z1, x2, z2 = electrode
    def term(xi, zi):
        return np.arctan(((xi - x)*(zi - z)) / (y * np.sqrt(y**2 + (xi - x)**2 + (zi - z)**2)))
    return (V / (2 * np.pi)) * (term(x2, z2) - term(x1, z2) - term(x2, z1) + term(x1, z1))

def potAll_rf(x, y, z, rf_ele_base, rf_ele_added, rf_ele_added_base):
    total = 0
    for ele in rf_ele_base:
        total += pot(x, y, z, ele, Vrf)
    for ele in rf_ele_added:
        total += pot(x, y, z, ele, Vrf)
    for ele in rf_ele_added_base:
        total += pot(x, y, z, ele, Vrf)
    return total

def pseud_pot(x, y, z, rf_ele_base, rf_ele_added, rf_ele_added_base):
    dx = dy = dz = 1e-6
    grad2 = (
        ((potAll_rf(x+dx, y, z, rf_ele_base, rf_ele_added, rf_ele_added_base) -
          potAll_rf(x-dx, y, z, rf_ele_base, rf_ele_added, rf_ele_added_base)) / (2*dx))**2 +
        ((potAll_rf(x, y+dy, z, rf_ele_base, rf_ele_added, rf_ele_added_base) -
          potAll_rf(x, y-dy, z, rf_ele_base, rf_ele_added, rf_ele_added_base)) / (2*dy))**2 +
        ((potAll_rf(x, y, z+dz, rf_ele_base, rf_ele_added, rf_ele_added_base) -
          potAll_rf(x, y, z-dz, rf_ele_base, rf_ele_added, rf_ele_added_base)) / (2*dz))**2
    )
    return pseud_A * grad2

# ===================================================
# :three: GA definition of symmetory electrode
# ===================================================
pq_starts = [(95 - 10 * i, 95 - 10 * i) for i in range(8)]
pqb_starts = [(975, 125 + 50 * i) for i in range(6)]
IND_SIZE = len(pq_starts) + len(pqb_starts)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def generate_monotonic_individual():
    small = sorted([random.randint(0, 50) for _ in range(len(pq_starts))], reverse=True)
    large = sorted([random.randint(0, 17 - i) for i in range(len(pqb_starts))], reverse=True)
    return creator.Individual(small + large)

toolbox.register("individual", generate_monotonic_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    n_add_small = ind[:len(pq_starts)]
    n_add_large = ind[len(pq_starts):]
    base_rf, added_rf, added_rf_base = generate_rf_electrodes(pq_starts, n_add_small, pqb_starts, n_add_large)
    rf_ele_base = np.array(extract_bounds(base_rf)) * um
    rf_ele_added = np.array(extract_bounds(added_rf)) * um
    rf_ele_added_base = np.array(extract_bounds(added_rf_base)) * um

    xpos = np.linspace(0, 500, 20) * um
    ypos = np.linspace(100, 300, 20) * um
    trap_heights = []
    for x in xpos:
        pots = [pseud_pot(x, y, 0, rf_ele_base, rf_ele_added, rf_ele_added_base) for y in ypos]
        trap_heights.append(ypos[np.argmin(pots)] / um)
    return np.mean([abs(y - 200) for y in trap_heights]),

def custom_mate(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    ind1[:len(pq_starts)] = sorted(ind1[:len(pq_starts)], reverse=True)
    ind1[len(pq_starts):] = sorted(ind1[len(pq_starts):], reverse=True)
    ind2[:len(pq_starts)] = sorted(ind2[:len(pq_starts)], reverse=True)
    ind2[len(pq_starts):] = sorted(ind2[len(pq_starts):], reverse=True)
    return ind1, ind2

def custom_mutate(ind):
    tools.mutUniformInt(ind, low=0, up=50, indpb=0.2)
    ind[:len(pq_starts)] = sorted(ind[:len(pq_starts)], reverse=True)
    ind[len(pq_starts):] = sorted(ind[len(pq_starts):], reverse=True)
    return ind,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
# give_intial_individualds
initial_ind = [44, 18, 12, 9, 6, 6, 2, 0, 18, 17, 16, 14, 5, 3]
population = [creator.Individual(initial_ind)] + toolbox.population(n=20)  # initial_indvidual1 + random_individual
NGEN = 1

for gen in tqdm(range(NGEN), desc="GA Progress"):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    best = tools.selBest(population, 1)[0]
    print(f"Generation {gen+1}: Best fitness = {best.fitness.values[0]:.4f}")

best_ind = tools.selBest(population, 1)[0]
print("Best individual:", best_ind)
print("Fitness:", best_ind.fitness.values[0])
#50-40-100-10(y精度細かく)-20-200

# Recreate_optimized_electrode
n_add_small = best_ind[:len(pq_starts)]
n_add_large = best_ind[len(pqb_starts):]
base_rf_final, added_rf_final, added_rf_base_final = generate_rf_electrodes(pq_starts, n_add_small, pqb_starts, n_add_large)

#---保存先---
folder_name = f"{initial_ind}_gen{NGEN}"
new_dir_path = os.path.join("/Users/miyamotomanari/Documents/PythonScripts/electrode_optimization/opt_add_rf/opt_file", folder_name)
os.makedirs(new_dir_path, exist_ok = True)

file_name = f"_gen{NGEN}_" + "_".join(map(str, best_ind)) + f"_f{best_ind.fitness.values[0]:.4f}"


# ===================================================
# :four: Draw_electrode
# ===================================================
fig, ax = plt.subplots(figsize=(8, 8))
for poly in base_rf_final:
    ax.add_patch(Polygon(poly, closed=True, facecolor='salmon', edgecolor='salmon'))
for poly in added_rf_final + added_rf_base_final:
    ax.add_patch(Polygon(poly, closed=True, facecolor='orangered', edgecolor='orangered'))
ax.set_aspect('equal')
ax.set_xlim(-6000, 6000)
ax.set_ylim(-6000, 6000)
plt.title("Optimized RF Electrode Layout (Full)")
plt.grid(True)
file_path = os.path.join(new_dir_path, 'RF_Layout' + file_name + '.png')
plt.savefig(file_path, dpi=300, transparent=True)
file_path = os.path.join(new_dir_path, 'RF_Layout' + file_name + '.svg')
plt.savefig(file_path, dpi=300, transparent=True)

fig, ax = plt.subplots(figsize=(8, 8))
for poly in base_rf_final:
    ax.add_patch(Polygon(poly, closed=True, facecolor='salmon', edgecolor='black'))
for poly in added_rf_final + added_rf_base_final:
    ax.add_patch(Polygon(poly, closed=True, facecolor='orangered', edgecolor='black'))
ax.set_aspect('equal')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
plt.title("Optimized RF Electrode Layout (Zoomed)")
plt.grid(True)
file_path = os.path.join(new_dir_path, 'RF_Layout_1000' + file_name + '.png')
plt.savefig(file_path, dpi=300, transparent=True)
file_path = os.path.join(new_dir_path, 'RF_Layout_1000' + file_name + '.svg')
plt.savefig(file_path, dpi=300, transparent=True)


# ===================================================
# :five: potential_color_map_and_height
# ===================================================
x_range = np.linspace(0, 2000, 100) * um
y_range = np.linspace(100, 300, 200) * um
X, Y = np.meshgrid(x_range, y_range)

rf_ele_base = np.array(extract_bounds(base_rf_final)) * um
rf_ele_added = np.array(extract_bounds(added_rf_final)) * um
rf_ele_added_base = np.array(extract_bounds(added_rf_base_final)) * um

Z = np.zeros_like(X)
trap_ys = []

for i in tqdm(range(len(x_range)), desc="Calculating potential map"):
    for j in range(len(y_range)):
        Z[j, i] = pseud_pot(x_range[i], y_range[j], 0, rf_ele_base, rf_ele_added, rf_ele_added_base)
    trap_y = y_range[np.argmin(Z[:, i])]
    trap_ys.append(trap_y / um)

plt.figure(figsize=(10, 6))
plt.contourf(x_range / um, y_range / um, Z, levels=50, cmap='viridis')
plt.colorbar(label="Pseudopotential [arb. unit]")
plt.plot(x_range / um, trap_ys, color='red', label='Trap Position')
plt.xlabel("X [µm]")
plt.ylabel("Y [µm]")
plt.title("Pseudopotential Map with Trap Position")
plt.legend()
plt.tight_layout()
file_path = os.path.join(new_dir_path, 'potential_map' + file_name + '.png')
plt.savefig(file_path, dpi=300, transparent=True)
file_path = os.path.join(new_dir_path, 'potential_mapt' + file_name + '.svg')
plt.savefig(file_path, dpi=300, transparent=True)

plt.figure(figsize=(8, 4))
plt.plot(x_range / um, trap_ys, marker='o', markersize=2)
plt.axhline(200, color='gray', linestyle='--', label='Target height 200 µm')
plt.xlabel("X [µm]")
plt.ylabel("Trap Height Y [µm]")
plt.title("Ion Trap Height Along X-axis")
plt.grid(True)
plt.legend()
plt.tight_layout()
file_path = os.path.join(new_dir_path, 'Height_curve' + file_name + '.png')
plt.savefig(file_path, dpi=300, transparent=True)
file_path = os.path.join(new_dir_path, 'Height_curve' + file_name + '.svg')
plt.savefig(file_path, dpi=300, transparent=True)