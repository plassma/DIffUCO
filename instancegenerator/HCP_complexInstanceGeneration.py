# Author: Veronika Semmelrock
# Created: 30.10.2025
# Description: Generator for satisfiable HCP instances (complex SAT-by-design)

"""
Complex SAT-by-design HCP instance generator (pure Python)
----------------------------------------------------------
Creates satisfiable instances with:
- person/1
- thing/1
- personTOthing/2
- cabinetDomain/1
- roomDomain/1

All constraints are satisfiable by construction:
- max 5 things per cabinet
- max 4 cabinets per room
- no mixed ownership
- ordering constraint satisfied
- optional gaps in thing IDs
"""

import random
from pathlib import Path


# DO NOT CHANGE - necessary for worst case upper bound of cabinet domain and room domain 
MAX_THINGS_PER_PERSON = 10
MAX_BLOCK_PER_PERSON = 3
NUM_CABINETS_PER_PERSON = 4  # see reasoning in README
NUM_ROOMS_PER_PERSON = 1     # see reasoning in README
MAX_PER_CABINET = 5
MAX_CABINETS_PER_ROOM = 4

# -------------------------
# Configurable parameters
# -------------------------
min_things_per_person = 2 # below MAX_THINGS_PER_PERSON! 
start_persons = 5
end_persons = 100
step = 5

# Generation behavior
allow_gaps = True
max_gap_size = 3
multi_block_prob = 0.25
seed = None  # set integer for reproducibility

# Output folder
script_dir = Path(__file__).parent
output_folder = script_dir / "generated_instances_complexSAT"
output_folder.mkdir(parents=True, exist_ok=True)

counter = 1
if seed is not None:
    random.seed(seed)

# -------------------------
# Helper functions
# -------------------------
def generate_segments_for_people(num_persons, min_tpp, multi_block_prob):
    """Create contiguous segments of things for each person."""
    counts = [random.randint(min_tpp, MAX_THINGS_PER_PERSON) for _ in range(num_persons)]
    blocks_per_person = []
    for c in counts:
        if c == 0:
            blocks_per_person.append(0)
            continue
        if c == 1:
            # Only one thing → at most one block
            blocks_per_person.append(1)
            continue
        if random.random() < multi_block_prob:
            # Use min(MAX_BLOCK_PER_PERSON, c): a person with only 2 things CANNOT be split into 3 blocks.
            b = random.randint(2, min(MAX_BLOCK_PER_PERSON, c))
        else:
            b = 1
        blocks_per_person.append(b)

    # Partition counts into blocks per person
    person_blocks = {}
    for pid, (count, blocks) in enumerate(zip(counts, blocks_per_person), start=1):
        if blocks <= 1:
            person_blocks[pid] = [count]
            continue
        cuts = sorted(random.sample(range(1, count), blocks - 1))
        sizes = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [count - cuts[-1]]
        person_blocks[pid] = [s for s in sizes if s > 0]

    # Flatten and shuffle blocks (no attempt to avoid adjacency; randomness is fine)
    pool = [(pid, sz) for pid, sizes in person_blocks.items() for sz in sizes]
    random.shuffle(pool)
    return pool, counts


def assign_thing_ids_to_segments(segments, allow_gaps=True, max_gap=3, start_id=1):
    """Assign thing IDs to segments, possibly inserting gaps."""
    cur = start_id
    seg_assignments = []
    for idx, (pid, length) in enumerate(segments):
        if idx > 0 and allow_gaps:
            gap = random.randint(0, max_gap)
            cur += gap
        ids = list(range(cur, cur + length))
        seg_assignments.append((pid, ids))
        cur += length
    return seg_assignments


# -------------------------
# Main generation loop
# -------------------------
for num_persons in range(start_persons, end_persons + 1, step):
    output_file = output_folder / f"hcp_complexSAT_{counter}_{num_persons}p_{MAX_THINGS_PER_PERSON}tMax.lp"

    # --- Generate SAT-by-design structure ---
    segments, person_counts = generate_segments_for_people(
        num_persons,
        min_things_per_person,
        multi_block_prob=multi_block_prob,
    )
    seg_assigns = assign_thing_ids_to_segments(segments, allow_gaps, max_gap_size, 1)

    # Collect things per person
    person_to_things = {p: [] for p in range(1, num_persons + 1)}
    all_things = []
    for pid, ids in seg_assigns:
        person_to_things[pid].extend(ids)
        all_things.extend(ids)
    for p in person_to_things:
        person_to_things[p].sort()

    # --- Compute necessary cabinet and room domain sizes ---
    total_cabinets = num_persons * NUM_CABINETS_PER_PERSON
    total_rooms = num_persons * NUM_ROOMS_PER_PERSON  # use the constant as intended

    # --- Create ASP facts ---
    person_facts = [f"person({p})." for p in range(1, num_persons + 1)]
    thing_facts = [f"thing({t})." for t in sorted(all_things)]
    personthing_facts = [f"personTOthing({p},{t})." for p, ts in person_to_things.items() for t in ts]
    cabinet_domain_facts = [f"cabinetDomain({i})." for i in range(1, total_cabinets + 1)]
    room_domain_facts = [f"roomDomain({i})." for i in range(1, total_rooms + 1)]

    # --- Write output file ---
    with open(output_file, "w") as f:
        f.write("\n".join(person_facts) + "\n")
        f.write("\n".join(thing_facts) + "\n")
        f.write("\n".join(personthing_facts) + "\n")
        f.write("\n".join(cabinet_domain_facts) + "\n")
        f.write("\n".join(room_domain_facts) + "\n")

    print(
        f"Generated {output_file} "
        f"-> {num_persons} persons, {len(all_things)} things, "
        f"{total_cabinets} cabinets, {total_rooms} rooms "
        f"(gaps={'on' if allow_gaps else 'off'})"
    )
    counter += 1
