#!/usr/bin/env python3
from collections import deque, defaultdict
from copy import deepcopy
from itertools import product
import heapq
"""
Smart Multi-Agent Multi-Floor Search

Features:
- Multi-floor support with floor transition cost
- Greedy zone allocation
- Tracks zones visited and paths per agent
- Multi-start optimization (all agents same start or combinations)
- Hallways: '1', entrances: '2', potential starts: '3'
- Rooms: '4' and above
"""

################################################################################
# PARSING
################################################################################


def parse_grid():
    print("Enter grid rows (press enter on empty line to finish):")
    grid = []
    while True:
        try:
            line = input().rstrip()
        except EOFError:
            break
        if line == "":
            break
        grid.append(list(line))

    if not grid:
        raise ValueError("Grid cannot be empty")

    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("All rows must have equal length")

    return grid


################################################################################
# BASIC HELPERS
################################################################################


def find_all_potential_starts(grid):
    return [(y, x) for y, row in enumerate(grid) for x, c in enumerate(row)
            if c == '3']


def find_all_passable_cells(grid):
    return [(y, x) for y, row in enumerate(grid) for x, c in enumerate(row)
            if c in {'1', '2', '3'}]


def find_all_entrances(grid):
    return [(y, x) for y, row in enumerate(grid) for x, c in enumerate(row)
            if c == '2']


def passable_for_bfs(cell):
    return cell in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}


def bfs_path(grid, start, goal):
    if start == goal:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                if (ny, nx) not in parent and passable_for_bfs(grid[ny][nx]):
                    parent[(ny, nx)] = (y, x)
                    q.append((ny, nx))
    return None


################################################################################
# ROOM ZONE DETECTION
################################################################################


def detect_room_zones(grid):
    visited = set()
    zones = []
    zone_map = {}
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c.isdigit() and int(c) >= 4 and (y, x) not in visited:
                stack = [(y, x)]
                visited.add((y, x))
                tiles = []
                while stack:
                    cy, cx = stack.pop()
                    tiles.append((cy, cx))
                    zone_map[(cy, cx)] = len(zones)
                    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                            if (ny, nx) not in visited and grid[ny][
                                    nx].isdigit() and int(grid[ny][nx]) >= 4:
                                visited.add((ny, nx))
                                stack.append((ny, nx))
                zones.append(tiles)
    return zone_map, zones


################################################################################
# MULTI-FLOOR DIJKSTRA
################################################################################


def dijkstra_path_multifloor(floors, start, goal, floor_transition_cost=5):
    num_floors = len(floors)
    visited = set()
    heap = []
    heapq.heappush(heap, (0, start))
    parent = {start: None}

    while heap:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        f, y, x = node
        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        # intra-floor neighbors
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(floors[f]) and 0 <= nx < len(floors[f][0]):
                if floors[f][ny][nx] in {
                        '1', '2', '3', '4', '5', '6', '7', '8', '9'
                }:
                    neighbor = (f, ny, nx)
                    if neighbor not in visited:
                        heapq.heappush(heap, (cost + 2, neighbor))
                        if neighbor not in parent:
                            parent[neighbor] = node

        # floor transitions
        for df in [-1, 1]:
            nf = f + df
            if 0 <= nf < num_floors:
                if floors[nf][y][x] in {
                        '1', '2', '3', '4', '5', '6', '7', '8', '9'
                }:
                    neighbor = (nf, y, x)
                    if neighbor not in visited:
                        heapq.heappush(
                            heap, (cost + floor_transition_cost, neighbor))
                        if neighbor not in parent:
                            parent[neighbor] = node
    return None


def best_path_to_zone_multifloor(agent_pos, zone_tiles, floors):
    best_path = None
    best_tile = None
    best_len = None
    for tile in zone_tiles:
        path = dijkstra_path_multifloor(floors, agent_pos, tile)
        if path is None:
            continue
        L = len(path) - 1
        if best_len is None or L < best_len:
            best_len = L
            best_path = path
            best_tile = tile
    return best_tile, best_path


################################################################################
# MULTI-FLOOR SIMULATION
################################################################################


def simulate_multi_multifloor(floors,
                              num_agents,
                              agent_starts_2d,
                              verbose=True):
    agent_starts = []
    for pos in agent_starts_2d:
        if len(pos) == 2:
            agent_starts.append((0, pos[0], pos[1]))
        elif len(pos) == 3:
            agent_starts.append(tuple(pos))
        else:
            raise ValueError(f"Invalid start position: {pos}")

    # detect zones per floor
    floor_zones = []
    for f, grid in enumerate(floors):
        _, zones = detect_room_zones(grid)
        floor_zones.append([[(f, y, x) for y, x in zone] for zone in zones])

    # global zone dict
    zones_remaining = {}
    zone_idx_global = 0
    for zones in floor_zones:
        for zone in zones:
            zones_remaining[zone_idx_global] = zone
            zone_idx_global += 1

    # init agents
    agents = []
    for i in range(num_agents):
        agents.append({
            'id': i + 1,
            'pos': agent_starts[i],
            'start_pos': agent_starts[i],
            'time': 0,
            'zones_visited': [],
            'path': [agent_starts[i]]
        })

    round_idx = 0
    if verbose:
        print(f"=== MULTI-FLOOR SIMULATION START ===")

    while zones_remaining:
        round_idx += 1
        if verbose:
            print(f"--- ROUND {round_idx} ---")

        candidates = []
        for a_idx, agent in enumerate(agents):
            for z_idx, tiles in zones_remaining.items():
                target_tile, path = best_path_to_zone_multifloor(
                    agent['pos'], tiles, floors)
                if path is None:
                    continue
                travel_cost = 0
                for j in range(1, len(path)):
                    f0, y0, x0 = path[j - 1]
                    f1, y1, x1 = path[j]
                    travel_cost += 5 if f0 != f1 else 2
                search_cost = len(tiles)
                total_cost = travel_cost + search_cost
                candidates.append((total_cost, a_idx, z_idx, target_tile, path,
                                   travel_cost, search_cost))

        if not candidates:
            if verbose:
                print("No reachable remaining zones. Stopping.")
            break

        candidates.sort(key=lambda t: t[0])
        assigned_agents = set()
        assigned_zones = set()
        assignments = []

        for total_cost, a_idx, z_idx, target_tile, path, travel_cost, search_cost in candidates:
            if a_idx in assigned_agents or z_idx in assigned_zones:
                continue
            assigned_agents.add(a_idx)
            assigned_zones.add(z_idx)
            assignments.append(
                (a_idx, z_idx, target_tile, path, travel_cost, search_cost))
            if len(assigned_agents) == len(agents):
                break

        for a_idx, z_idx, target_tile, path, travel_cost, search_cost in assignments:
            agent = agents[a_idx]
            agent['pos'] = path[-1]
            agent['time'] += travel_cost + search_cost
            agent['zones_visited'].append(z_idx)
            agent['path'].extend(path[1:])
            if verbose:
                print(
                    f"Agent {agent['id']} -> Zone {z_idx} (floor {agent['pos'][0]}) travel:{travel_cost}s search:{search_cost}s total:{agent['time']}s"
                )
            del zones_remaining[z_idx]

    # return to ground floor start
    if verbose:
        print("=== RETURNING TO GROUND FLOOR STARTS ===")
    for agent in agents:
        start = agent['start_pos']
        path = dijkstra_path_multifloor(floors, agent['pos'], start)
        if path:
            return_cost = 0
            for j in range(1, len(path)):
                f0, y0, x0 = path[j - 1]
                f1, y1, x1 = path[j]
                return_cost += 5 if f0 != f1 else 2
            agent['time'] += return_cost
            agent['pos'] = start
            agent['path'].extend(path[1:])
            if verbose:
                print(
                    f"Agent {agent['id']} returned to start. Cost:{return_cost}s total:{agent['time']}s"
                )
        else:
            if verbose:
                print(f"Agent {agent['id']} cannot return to start!")

    makespan = max(agent['time'] for agent in agents)
    if verbose:
        print("=== SIMULATION SUMMARY ===")
        for agent in agents:
            print(
                f"Agent {agent['id']} time:{agent['time']}s zones:{agent['zones_visited']}"
            )
    return makespan, agents


################################################################################
# MULTI-FLOOR OPTIMAL START TESTING
################################################################################


def find_optimal_start_multifloor(floors,
                                  num_agents,
                                  start_positions,
                                  test_combinations=False):
    if test_combinations:
        all_combos = list(product(start_positions, repeat=num_agents))
        print(f"Testing {len(all_combos)} combinations...")
        results = {}
        for idx, combo in enumerate(all_combos, 1):
            print(f"Test {idx}/{len(all_combos)}: {combo}")
            makespan, _ = simulate_multi_multifloor(floors,
                                                    num_agents,
                                                    combo,
                                                    verbose=False)
            results[combo] = makespan
            print(f"  Makespan: {makespan}s\n")
        best_config = min(results, key=results.get)
        print(
            f"Best configuration found: {best_config}, makespan={results[best_config]}s\n"
        )
        return best_config, results[best_config], results
    else:
        results = {}
        for idx, pos in enumerate(start_positions, 1):
            print(f"Test {idx}/{len(start_positions)}: all agents at {pos}")
            makespan, _ = simulate_multi_multifloor(floors,
                                                    num_agents,
                                                    [pos] * num_agents,
                                                    verbose=False)
            results[(pos, ) * num_agents] = makespan
            print(f"  Makespan: {makespan}s\n")
        best_config = min(results, key=results.get)
        print(
            f"Best configuration found: {best_config}, makespan={results[best_config]}s\n"
        )
        return best_config, results[best_config], results


################################################################################
# MAIN
################################################################################


def main():
    print("=== MULTI-FLOOR SMART AGENT SIMULATOR ===")
    num_floors = int(input("Number of floors: ").strip() or "1")
    floors = []
    for f in range(num_floors):
        print(f"Enter grid for floor {f} (empty line to finish):")
        floors.append(parse_grid())

    num_agents = int(input("Number of agents (>=1): ").strip() or "1")
    num_agents = max(1, num_agents)

    # Starting positions
    start_positions = []
    print(
        "Specify starting positions (y,x) on ground floor (empty line to finish):"
    )
    while True:
        line = input().strip()
        if line == "": break
        y, x = map(int, line.split(','))
        start_positions.append((y, x))
    if not start_positions:
        print("No starting positions. Using all '3's on ground floor.")
        start_positions = find_all_potential_starts(floors[0])
    print(f"{len(start_positions)} starting positions found.")

    # Optimization choice
    test_combinations = False
    if len(start_positions) > 1 and num_agents > 1:
        choice = input(
            "Test all combinations for optimal start? (y/n, default n): "
        ).strip().lower()
        test_combinations = (choice == 'y')

    best_config, best_makespan, _ = find_optimal_start_multifloor(
        floors, num_agents, start_positions, test_combinations)
    print(
        f"Best start configuration: {best_config}, makespan={best_makespan}s")

    # Run simulation with best configuration and verbose
    makespan, agents_info = simulate_multi_multifloor(floors,
                                                      num_agents,
                                                      best_config,
                                                      verbose=True)
    for agent in agents_info:
        print(f"Agent {agent['id']} visited zones {agent['zones_visited']}")
        print(f"Path: {agent['path']}")


if __name__ == '__main__':
    main()
