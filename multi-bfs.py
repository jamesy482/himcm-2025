#!/usr/bin/env python3
from collections import deque, defaultdict
from copy import deepcopy
from itertools import product
"""
Smart Multi-Agent Search with Multi-Start Optimization (Multi-Floor)

Notes:
- Floors are entered floor-by-floor.
- Floor coordinate ordering is (f, y, x) where f is floor index (0..F-1).
- Hallways: '1' (walkable), entrances: '2' (walkable), potential starts: '3' (walkable)
- Stairs: 'S' : vertical connector — must be 'S' on both floors at same (y,x) to move up/down.
- Rooms: digits '4' and above. Each connected cluster (per-floor) is a zone.
- Global room search: once a zone is searched, it's marked globally.
- Smart task allocation: greedy assignment of agents to zones each round.
- Movement costs: horizontal step = 2s, vertical stair step = 5s. Room search cost = 1s per tile.
- Return-to-start cost: same movement rules applied, computed from BFS path.
"""

################################################################################
# PARSING MULTI-FLOOR INPUT
################################################################################


def parse_floors():
    """
    Ask user how many floors, then read each floor grid row-by-row.
    Finish each floor by entering an empty line. Floors must all have the same row length
    individually (different floors may have different sizes, but that's discouraged).
    Returns list_of_floors, each a list of list of chars.
    """
    try:
        nf = int(input("Number of floors (>=1): ").strip())
    except Exception:
        nf = 1
    if nf < 1:
        nf = 1

    floors = []
    for fi in range(nf):
        print(f"\nEnter rows for FLOOR {fi} (empty line to finish this floor):")
        grid = []
        while True:
            try:
                line = input().rstrip("")
            except EOFError:
                break
            if line == "":
                break
            grid.append(list(line))
        if not grid:
            print(f"Warning: Floor {fi} is empty. Adding a single 0 tile.")
            grid = [['0']]
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValueError(f"All rows in floor {fi} must have equal length")
        floors.append(grid)

    return floors


################################################################################
# BASIC HELPERS (3D-aware)
################################################################################


def find_all_potential_starts(floors):
    """Find all '3' cells across floors. Returns list of (f,y,x)."""
    starts = []
    for f, grid in enumerate(floors):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == '3':
                    starts.append((f, y, x))
    return starts


def find_all_passable_cells(floors):
    """Find all passable cells (1,2,3,S) across floors. Returns list of (f,y,x)."""
    passable = []
    for f, grid in enumerate(floors):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c in {'1', '2', '3', 'S'}:
                    passable.append((f, y, x))
    return passable


def find_all_entrances(floors):
    ents = []
    for f, grid in enumerate(floors):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == '2':
                    ents.append((f, y, x))
    return ents


def passable_for_bfs(cell):
    """
    Which tile characters are allowed to be traversed horizontally or used as S anchors.
    We allow '1','2','3','S' and room digits '4'..'9' (keeps previous behavior of allowing rooms).
    """
    return cell in {'1', '2', '3', 'S', '4', '5', '6', '7', '8', '9'}


def in_bounds_floor(floors, f, y, x):
    if 0 <= f < len(floors) and 0 <= y < len(floors[f]) and 0 <= x < len(floors[f][0]):
        return True
    return False


def bfs_path(floors, start, goal):
    """
    BFS in 3D (floor, y, x) with vertical moves allowed only between 'S' tiles at same (y,x).
    Returns path as list of (f,y,x) or None.
    """
    if start == goal:
        return [start]

    q = deque([start])
    parent = {start: None}

    while q:
        f, y, x = q.popleft()
        if (f, y, x) == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        # horizontal neighbors on same floor
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny = y + dy
            nx = x + dx
            nf = f
            if in_bounds_floor(floors, nf, ny, nx):
                if (nf, ny, nx) not in parent and passable_for_bfs(floors[nf][ny][nx]):
                    parent[(nf, ny, nx)] = (f, y, x)
                    q.append((nf, ny, nx))

        # vertical neighbors via 'S' only (up/down)
        # move up
        if floors[f][y][x] == 'S':
            upf = f + 1
            if upf < len(floors) and in_bounds_floor(floors, upf, y, x):
                if floors[upf][y][x] == 'S' and (upf, y, x) not in parent:
                    parent[(upf, y, x)] = (f, y, x)
                    q.append((upf, y, x))
            # move down
            downf = f - 1
            if downf >= 0 and in_bounds_floor(floors, downf, y, x):
                if floors[downf][y][x] == 'S' and (downf, y, x) not in parent:
                    parent[(downf, y, x)] = (f, y, x)
                    q.append((downf, y, x))

    return None


################################################################################
# ROOM ZONE DETECTION and CONNECTIONS (multi-floor)
################################################################################


def detect_room_zones(floors):
    """
    Detect connected room zones per-floor (digit >=4). Zones are given global indices.
    Returns:
      zone_map: dict {(f,y,x): zone_idx}
      zones: list of lists of (f,y,x) tiles for each zone (global index -> tiles)
    """
    visited = set()
    zones = []
    zone_map = {}

    for f, grid in enumerate(floors):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c.isdigit() and int(c) >= 4 and (f, y, x) not in visited:
                    stack = [(f, y, x)]
                    visited.add((f, y, x))
                    tiles = []
                    while stack:
                        cf, cy, cx = stack.pop()
                        tiles.append((cf, cy, cx))
                        zone_map[(cf, cy, cx)] = len(zones)
                        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            ny, nx = cy + dy, cx + dx
                            nf = cf
                            if in_bounds_floor(floors, nf, ny, nx):
                                if (nf, ny, nx) not in visited:
                                    if floors[nf][ny][nx].isdigit() and int(
                                            floors[nf][ny][nx]) >= 4:
                                        visited.add((nf, ny, nx))
                                        stack.append((nf, ny, nx))
                    zones.append(tiles)

    return zone_map, zones


def detect_zone_connections_via_2(floors, zone_map):
    """
    Build adjacency between zones via '2' tiles on the same floor.
    Returns connections dict and connector_map mapping zone-pair -> list of entrance (f,y,x) coordinates connecting them.
    """
    connections = defaultdict(set)
    connector_map = defaultdict(list)  # (zoneA, zoneB) -> list of (f,y,x)

    for f, grid in enumerate(floors):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == '2':
                    adjacent_zones = set()
                    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ny, nx = y + dy, x + dx
                        if in_bounds_floor(floors, f, ny, nx):
                            if (f, ny, nx) in zone_map:
                                adjacent_zones.add(zone_map[(f, ny, nx)])
                    adj_list = list(adjacent_zones)
                    for i in range(len(adj_list)):
                        for j in range(i + 1, len(adj_list)):
                            a, b = adj_list[i], adj_list[j]
                            connections[a].add(b)
                            connections[b].add(a)
                            connector_map[tuple(sorted((a, b)))].append((f, y, x))

    return connections, connector_map


def entrances_for_zones(floors, entrances, zone_map):
    """
    Map each entrance (f,y,x) to the adjacent zone index (or None) on the same floor.
    Returns mapping entrance -> zone_idx or None.
    """
    ent_zone = {}
    for e in entrances:
        f, y, x = e
        z = None
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ny, nx = y + dy, x + dx
            if in_bounds_floor(floors, f, ny, nx):
                if (f, ny, nx) in zone_map:
                    z = zone_map[(f, ny, nx)]
                    break
        ent_zone[e] = z
    return ent_zone


################################################################################
# PATH VISUALIZATION (multi-floor)
################################################################################


def mark_floors_path(floors, path):
    """
    Return a deep-copied list of floor grids with '*' marking where path visits (excluding walls '0',
    and preserving '2','3','S' and room digits).
    """
    floors_copy = [deepcopy(grid) for grid in floors]
    for f, y, x in path:
        if not in_bounds_floor(floors_copy, f, y, x):
            continue
        c = floors_copy[f][y][x]
        if c != '0':
            # Avoid overwriting '2','3','S', and room digits
            if c not in {'2', '3', 'S'} and not (c.isdigit() and int(c) >= 4):
                floors_copy[f][y][x] = '*'
    return floors_copy


def print_floors(floors):
    for f, grid in enumerate(floors):
        print(f"--- FLOOR {f} ---")
        for row in grid:
            print(''.join(row))


################################################################################
# SMART TASK ALLOCATION LOOP (multi-floor)
################################################################################


def simulate_multi(floors, num_agents, start_positions, verbose=True):
    """
    Multi-floor simulation. start_positions: either (f,y,x) for all agents, or list/tuple of positions (one per agent).
    Returns makespan or None on failure.
    """
    # Normalize start_positions
    if isinstance(start_positions, tuple) and len(start_positions) == 3 and isinstance(start_positions[0], int):
        agent_starts = [start_positions] * num_agents
    else:
        agent_starts = list(start_positions)
        if len(agent_starts) != num_agents:
            if verbose:
                print(
                    f"ERROR: Number of start positions ({len(agent_starts)}) doesn't match number of agents ({num_agents})"
                )
            return None

    entrances = find_all_entrances(floors)
    if not entrances:
        if verbose:
            print("ERROR: No entrances ('2') found in floors.")
        return None

    zone_map, zones = detect_room_zones(floors)
    connections, connector_map = detect_zone_connections_via_2(floors, zone_map)
    ent_zone = entrances_for_zones(floors, entrances, zone_map)

    if verbose:
        print(f"Detected {len(zones)} zones")
        if connections:
            print(
                f"Detected zone connections via '2' tiles: {dict(connections)}"
            )

    zones_remaining = set(range(len(zones)))

    # Agents
    agents = []
    for i in range(num_agents):
        agents.append({
            'id': i + 1,
            'pos': agent_starts[i],
            'start_pos': agent_starts[i],
            'time': 0,
            'zones_visited': []
        })

    round_idx = 0

    if verbose:
        print('=== SIMULATION START ===')
        if len(set(agent_starts)) == 1:
            print(f"All agents start at: {agent_starts[0]}")
        else:
            print("Agent starting positions:")
            for i, pos in enumerate(agent_starts):
                print(f"  Agent {i+1}: {pos}")
        print(f"Entrances: {sorted(entrances)}")

    # Helper: compute best entrance and path for an agent to reach a zone
    def best_path_to_zone(agent_pos, zone_idx):
        candidates = [e for e, z in ent_zone.items() if z == zone_idx]
        best_path = None
        best_ent = None
        best_cost_metric = None  # prefer fewer steps then tie-breaker
        for ent in candidates:
            path = bfs_path(floors, agent_pos, ent)
            if path is None:
                continue
            # compute travel cost along path: horizontal steps=2, vertical (floor change)=5
            travel_cost = 0
            for i in range(1, len(path)):
                (f0, y0, x0) = path[i - 1]
                (f1, y1, x1) = path[i]
                if f0 != f1:
                    travel_cost += 5
                else:
                    travel_cost += 2
            # Use travel_cost as metric (smaller is better). We'll return path and entrance.
            if best_cost_metric is None or travel_cost < best_cost_metric:
                best_cost_metric = travel_cost
                best_path = path
                best_ent = ent
        return best_ent, best_path

    # Main greedy loop
    while zones_remaining:
        round_idx += 1
        if verbose:
            print(f"--- ROUND {round_idx} ---")

        candidates = []
        for a_idx, agent in enumerate(agents):
            for z in zones_remaining:
                ent, path = best_path_to_zone(agent['pos'], z)
                if path is None:
                    continue
                # compute travel_cost (detailed)
                travel_cost = 0
                for i in range(1, len(path)):
                    f0, y0, x0 = path[i - 1]
                    f1, y1, x1 = path[i]
                    if f0 != f1:
                        travel_cost += 5
                    else:
                        travel_cost += 2
                search_cost = len(zones[z])
                total_cost = travel_cost + search_cost
                candidates.append(
                    (total_cost, a_idx, z, ent, path, travel_cost,
                     search_cost))

        if not candidates:
            if verbose:
                print("No reachable remaining zones by any agent. Stopping.")
            break

        candidates.sort(key=lambda t: t[0])

        assigned_agents = set()
        assigned_zones = set()
        assignments = []

        for total_cost, a_idx, z, ent, path, travel_cost, search_cost in candidates:
            if a_idx in assigned_agents or z in assigned_zones:
                continue
            assigned_agents.add(a_idx)
            assigned_zones.add(z)
            assignments.append((a_idx, z, ent, path, travel_cost, search_cost))
            if len(assigned_agents) == len(agents):
                break

        for a_idx, z, ent, path, travel_cost, search_cost in assignments:
            agent = agents[a_idx]
            agent['time'] += travel_cost
            agent['pos'] = ent

            if verbose:
                print(f"Agent {agent['id']} -> Entrance {ent} (Zone {z})")
                print(
                    f"  Travel steps: {len(path) - 1} (+{travel_cost}s). Agent time now: {agent['time']}s"
                )
                print("  Path:")
                print_floors(mark_floors_path(floors, path))

            agent['time'] += search_cost
            if verbose:
                print(
                    f"  Searched Zone {z} area={search_cost} (+{search_cost}s). Agent time now: {agent['time']}s"
                )
            agent['zones_visited'].append(z)
            if z in zones_remaining:
                zones_remaining.remove(z)

        if not assignments:
            if verbose:
                print(
                    "No assignments could be made this round. Stopping to avoid infinite loop."
                )
            break

    # Return to starts
    if verbose:
        print('=== RETURNING TO START ===')
    for agent in agents:
        start = agent['start_pos']
        path = bfs_path(floors, agent['pos'], start)
        if path is None:
            if verbose:
                print(
                    f"Agent {agent['id']} cannot return to start from {agent['pos']}!"
                )
            continue
        # compute return cost
        cost = 0
        for i in range(1, len(path)):
            f0, y0, x0 = path[i - 1]
            f1, y1, x1 = path[i]
            if f0 != f1:
                cost += 5
            else:
                cost += 2
        agent['time'] += cost
        agent['pos'] = start

        if verbose:
            print(f"Agent {agent['id']} returning to start at {start}")
            print(
                f"  Return steps: {len(path) - 1} (+{cost}s). Agent time now: {agent['time']}s"
            )
            print_floors(mark_floors_path(floors, path))

    if verbose:
        print('=== SUMMARY ===')
        for agent in agents:
            visited = agent['zones_visited']
            print(f"Agent {agent['id']}: total time = {agent['time']}s")
            print(f"  Zones visited: {visited}")
    makespan = max(agent['time'] for agent in agents) if agents else 0
    if verbose:
        print(f"Makespan (slowest agent): {makespan}s")

    return makespan


################################################################################
# OPTIMIZATION (multi-floor aware)
################################################################################


def find_optimal_start(floors,
                       num_agents,
                       start_positions,
                       test_combinations=False):
    """
    Test multiple starting positions and find the most efficient one.
    start_positions: list of (f,y,x) tuples to test
    """
    if test_combinations:
        all_combinations = list(product(start_positions, repeat=num_agents))
        print(f"\n{'=' * 60}")
        print(f"TESTING ALL COMBINATIONS OF STARTING POSITIONS")
        print(
            f"{len(start_positions)} positions × {num_agents} agents = {len(all_combinations)} combinations"
        )
        print(f"{'=' * 60}\n")

        if len(all_combinations) > 100:
            print(
                f"WARNING: Testing {len(all_combinations)} combinations may take a while."
            )
            try:
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    return None, None, {}
            except:
                pass

        results = {}

        for idx, combo in enumerate(all_combinations):
            print(f"\n{'*' * 60}")
            print(f"TEST {idx + 1}/{len(all_combinations)}: Configuration")
            for agent_idx, pos in enumerate(combo):
                print(f"  Agent {agent_idx + 1}: {pos}")
            print(f"{'*' * 60}\n")

            makespan = simulate_multi(floors, num_agents, combo, verbose=True)

            if makespan is not None:
                results[combo] = makespan
            else:
                print(f"Simulation failed for configuration {combo}")

        if not results:
            print("\nERROR: No valid simulations completed.")
            return None, None, {}

        best_config = min(results, key=results.get)
        best_makespan = results[best_config]

        print(f"\n{'=' * 60}")
        print("COMPARISON OF ALL CONFIGURATIONS")
        print(f"{'=' * 60}\n")

        sorted_results = sorted(results.items(), key=lambda x: x[1])
        display_count = min(20, len(sorted_results))
        print(f"Showing top {display_count} configurations:\n")

        for config, makespan in sorted_results[:display_count]:
            marker = " ← BEST" if config == best_config else ""
            config_str = ", ".join(
                [f"A{i+1}:{pos}" for i, pos in enumerate(config)])
            print(f"{config_str}: {makespan}s{marker}")

        if len(sorted_results) > display_count:
            print(
                f"\n... and {len(sorted_results) - display_count} more configurations"
            )

        print(f"\n{'=' * 60}")
        print(f"OPTIMAL CONFIGURATION:")
        for i, pos in enumerate(best_config):
            print(f"  Agent {i+1}: {pos}")
        print(f"Best Makespan: {best_makespan}s")
        print(f"{'=' * 60}\n")

        return best_config, best_makespan, results

    else:
        print(f"\n{'=' * 60}")
        print(f"TESTING {len(start_positions)} STARTING POSITIONS")
        print(f"(All agents start at the same position)")
        print(f"{'=' * 60}\n")

        results = {}

        for idx, start_pos in enumerate(start_positions):
            print(f"\n{'*' * 60}")
            print(
                f"TEST {idx + 1}/{len(start_positions)}: Starting Position {start_pos}"
            )
            print(f"{'*' * 60}\n")

            makespan = simulate_multi(floors,
                                      num_agents,
                                      start_pos,
                                      verbose=True)

            if makespan is not None:
                config = tuple([start_pos] * num_agents)
                results[config] = makespan
            else:
                print(f"Simulation failed for starting position {start_pos}")

        if not results:
            print("\nERROR: No valid simulations completed.")
            return None, None, {}

        best_config = min(results, key=results.get)
        best_makespan = results[best_config]

        print(f"\n{'=' * 60}")
        print("COMPARISON OF ALL STARTING POSITIONS")
        print(f"{'=' * 60}\n")

        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for config, makespan in sorted_results:
            pos = config[0]
            marker = " ← BEST" if config == best_config else ""
            print(f"Position {pos}: {makespan}s{marker}")

        print(f"\n{'=' * 60}")
        print(f"OPTIMAL STARTING POSITION: {best_config[0]}")
        print(f"Best Makespan: {best_makespan}s")
        print(f"{'=' * 60}\n")

        return best_config, best_makespan, results


################################################################################
# MAIN
################################################################################


def main():
    floors = parse_floors()

    try:
        num_agents = int(input("Number of agents (>=1): ").strip())
    except Exception:
        num_agents = 1
    if num_agents < 1:
        num_agents = 1

    # Ask user how they want to specify starting positions
    print("\nHow do you want to specify starting positions?")
    print("1. Use all cells marked with '3' in the floors")
    print("2. Use all passable cells (1, 2, 3, S) as potential starts")
    print("3. Enter specific coordinates manually")

    try:
        choice = input("Enter choice (1-3, default=1): ").strip()
        if choice == "":
            choice = "1"
    except Exception:
        choice = "1"

    start_positions = []

    if choice == "1":
        start_positions = find_all_potential_starts(floors)
        if not start_positions:
            print("No '3' cells found. Falling back to manual entry.")
            choice = "3"
        else:
            print(
                f"Found {len(start_positions)} potential starting positions marked with '3':"
            )
            for pos in start_positions:
                print(f"  {pos}")

    if choice == "2":
        start_positions = find_all_passable_cells(floors)
        print(
            f"Found {len(start_positions)} passable cells to test as starting positions."
        )

        if len(start_positions) > 20:
            confirm = input(
                f"Warning: This will test {len(start_positions)} positions. Continue? (y/n): "
            ).strip().lower()
            if confirm != 'y':
                choice = "3"
                start_positions = []

    if choice == "3" or not start_positions:
        print("\nEnter starting positions as coordinates (f,y,x).")
        print("Example: 0,0,5 for floor 0 row 0 column 5")
        print("Enter one per line, empty line to finish:")

        while True:
            try:
                line = input().strip()
                if line == "":
                    break
                parts = line.split(',')
                if len(parts) != 3:
                    print("  Invalid format; use f,y,x")
                    continue
                f, y, x = int(parts[0]), int(parts[1]), int(parts[2])
                if in_bounds_floor(floors, f, y, x):
                    start_positions.append((f, y, x))
                    print(f"  Added position ({f}, {y}, {x})")
                else:
                    print(f"  Invalid position ({f}, {y}, {x}) - out of bounds")
            except Exception as e:
                print(f"  Invalid input: {e}")

    if not start_positions:
        print("ERROR: No starting positions specified. Exiting.")
        return

    if len(start_positions) == 1:
        print(f"\nRunning simulation with starting position {start_positions[0]}...")
        simulate_multi(floors, num_agents, start_positions[0], verbose=True)
    else:
        print("\nOptimization mode:")
        print("1. Test all agents starting at the same position (faster)")
        print(
            "2. Test all combinations where agents can start at different positions (slower but more optimal)"
        )

        try:
            opt_choice = input("Enter choice (1-2, default=1): ").strip()
            if opt_choice == "":
                opt_choice = "1"
        except Exception:
            opt_choice = "1"

        test_combinations = (opt_choice == "2")

        find_optimal_start(floors,
                           num_agents,
                           start_positions,
                           test_combinations=test_combinations)


if __name__ == '__main__':
    main()
