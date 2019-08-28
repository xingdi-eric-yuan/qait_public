import numpy as np
import random
import uuid
import os
import time
import multiprocessing as mp
from os.path import join as pjoin

with open("vocabularies/fake_words.txt") as f:
    FAKE_WORDS = f.read().lower().split("\n")
FAKE_WORDS = set(FAKE_WORDS)


# entities
with open("vocabularies/entities.txt") as f:
    ALL_ENTITIES = f.read().lower().split("\n")
ALL_ENTITIES = set(ALL_ENTITIES)


########################################################## question generator


def attribute_to_question(attr, object_name):
    if attr == "edible":
        res = "is " + object_name + " edible ?"
    elif attr == "drinkable":
        res = "is " + object_name + " drinkable ?"
    elif attr == "portable":
        res = "can one put " + object_name + " into pocket ?"
    elif attr == "openable":
        res = "is it possible to open " + object_name + " ?"
    elif attr == "cuttable":
        res = "can " + object_name + " be cut ?"
    elif attr == "sharp":
        res = "can one cut anything with a " + object_name + " ?"
    elif attr == "heat_source":
        res = "can one cook things with the " + object_name + " ?"
    elif attr == "cookable":
        res = "can the " + object_name + " be cooked ?"
    elif attr == "holder":
        res = "can " + object_name + " hold things ?"
    else:
        print("Unknown: {}".format(attr))
        raise NotImplementedError
    return res


def generate_location_question(entity_dict, seed=None):
    # entity_dict is a dict of {entity: location}
    entities, locations = [], []
    for item in entity_dict:
        if item == "" or entity_dict[item] == "":
            continue
        loc = entity_dict[item]
        item, loc = item.lower(), loc.lower()
        # use most immediate container as answer
        if "." in loc:
            loc = loc.rsplit(".")[-1]
        # filter out multi-word locations
        if " " in loc:
            continue
        entities.append(item)
        locations.append(loc)
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.randint(low=0, high=len(entities))
    return "where is the " + entities[idx] + " ?", locations[idx], entities[idx]


def generate_attribute_question(entity_dict, seed=None):
    # entity_dict is a dict of {entity: attribute}
    if seed is not None:
        np.random.seed(seed)

    all_attributes = set(["edible", "drinkable", "portable", "openable",
                          "cuttable", "sharp", "heat_source", "cookable",
                          "holder"])
    all_entities = set()
    attribute_dict = dict()
    for item in entity_dict:
        if item not in FAKE_WORDS:
            continue

        attrs_of_this_obj = list(set(entity_dict[item]) & all_attributes)
        for attr in attrs_of_this_obj:
            if attr not in attribute_dict:
                attribute_dict[attr] = set()
            attribute_dict[attr].add(item)
        all_entities.add(item)

    all_attributes = sorted([key for key in attribute_dict])
    random_attr = np.random.choice(all_attributes)
    entity_true = attribute_dict[random_attr]
    entity_false = sorted(all_entities - entity_true)
    entity_true = sorted(entity_true)

    if len(entity_false) == 0 or len(entity_true) == 0:
        assert False, "Contact Marc if this happens!"
        #if seed is not None:
        #    seed = seed + 1
        # return generate_attribute_question(entity_dict, seed)

    if np.random.rand() > 0.5:
        answer = "1"
        entity_ = np.random.choice(entity_true)
    else:
        answer = "0"
        entity_ = np.random.choice(entity_false)

    return attribute_to_question(random_attr, entity_), answer, random_attr, entity_


def generate_existence_question(entity_dict, seed=None):
    # entity_dict is a dict of {entity: location}
    entities_in_this_game = []
    for item in entity_dict:
        item = item.lower()
        if item == "" or entity_dict[item] == "":
            continue
        entities_in_this_game.append(item)
    entities_not_in_this_game = list(ALL_ENTITIES - set(entities_in_this_game) - FAKE_WORDS)

    if seed is not None:
        np.random.seed(seed)
    if np.random.rand() > 0.5:
        entity = np.random.choice(entities_in_this_game)
        return "is there any " + entity + " in the world ?", "1", entity
    else:
        entity = np.random.choice(entities_not_in_this_game)
        return "is there any " + entity + " in the world ?", "0", entity


def generate_qa_pairs(infos, question_type="location", seed=42):
    output_questions, output_answers = [], []
    reward_helper_info = {"batch_size": len(infos["extra.object_locations"]),
                          "_entities": [],
                          "_answers": [],
                          "_attributes": []}
    for i in range(len(infos["extra.object_locations"])):
        if question_type == "location":
            _q, _a, _e = generate_location_question(infos["extra.object_locations"][i], seed=seed * len(infos["extra.object_locations"]) + i)
        elif question_type == "attribute":
            _q, _a, _attr, _e = generate_attribute_question(infos["extra.object_attributes"][i], seed=seed * len(infos["extra.object_locations"]) + i)
            reward_helper_info["_attributes"].append(_attr)
        elif question_type == "existence":
            _q, _a, _e = generate_existence_question(infos["extra.object_locations"][i], seed=seed * len(infos["extra.object_locations"]) + i)
        else:
            raise NotImplementedError
        output_questions.append(_q)
        output_answers.append(_a)
        reward_helper_info["_entities"].append(_e)  # the entity being asked
        reward_helper_info["_answers"].append(_a)  # the entity being asked

    return output_questions, output_answers, reward_helper_info


########################################################## game generator


def generate_fixed_map_games(p_num, path="./", question_type="location", random_seed=None, num_object=None):
    if random_seed is None:
        np.random.seed()
    else:
        np.random.seed(random_seed)
    # generate fixed map games
    map_seed = 123
    num_room = 6
    if num_object is None:
        num_object = np.random.randint(low=num_room * 3, high=num_room * 6 + 1)
    if random_seed is None:
        random_seed = np.random.randint(100000000)
    with_placeholders = question_type == "attribute"
    random_game_name = str(uuid.uuid1()) + str(p_num)
    config_list = [str(num_room), str(num_object), str(map_seed), str(with_placeholders), str(random_seed)]
    random_game_name += "_config_" + "_".join(config_list)
    gamefile = pjoin(path, "game_" + random_game_name + ".ulx")

    cmd = "tw-make tw-iqa --nb-rooms " + str(num_room) + " --nb-entities " + str(num_object) + " --seed-map " + str(map_seed) + (" --with-placeholders" if with_placeholders else "") +\
        " --third-party challenge.py --seed " + str(random_seed) + " --output " + gamefile + " --silent --kb " + pjoin(path, "textworld_data")
    os.system(cmd)
    return gamefile


def generate_random_map_games(p_num, path="./", question_type="location", random_seed=None, num_room=None, num_object=None):
    if random_seed is None:
        np.random.seed()
    else:
        np.random.seed(random_seed)

    # generate random map games
    num_room_lower_bound = 2
    num_room_upper_bound = 12
    if num_room is None:
        num_room = np.random.randint(low=num_room_lower_bound, high=num_room_upper_bound + 1)

    with_placeholders = question_type == "attribute"
    if with_placeholders:
        num_room = max(num_room, 2)  # Placeholder option requires at least two rooms.

    if num_object is None:
        num_object = np.random.randint(low=num_room * 3, high=num_room * 6 + 1)

    if random_seed is None:
        random_seed = np.random.randint(100000000)

    map_seed = random_seed

    random_game_name = str(uuid.uuid1()) + str(p_num)
    config_list = [str(num_room), str(num_object), str(map_seed), str(with_placeholders), str(random_seed)]
    random_game_name += "_config_" + "_".join(config_list)
    gamefile = pjoin(path, "game_" + random_game_name + ".ulx")

    cmd = "tw-make tw-iqa --nb-rooms " + str(num_room) + " --nb-entities " + str(num_object) + " --seed-map " + str(map_seed) + (" --with-placeholders" if with_placeholders else "") +\
        " --third-party challenge.py --seed " + str(random_seed) + " --output " + gamefile + " --silent --kb " + pjoin(path, "textworld_data")
    os.system(cmd)
    return gamefile


def game_generator_queue(path="./", random_map=False, question_type="location", max_q_size=30, wait_time=0.5, nb_worker=1):

    q = mp.Queue()
    nb_worker = min(nb_worker, mp.cpu_count() - 1)

    def data_generator_task(p_num):
        counter = 0
        while True:
            np.random.seed(p_num * 12345 + counter)
            seed = np.random.randint(100000000)
            if q.qsize() < max_q_size:
                try:
                    if random_map:
                        game_file_name = generate_random_map_games(p_num, path=path, question_type=question_type, random_seed=seed)
                    else:
                        game_file_name = generate_fixed_map_games(p_num, path=path, question_type=question_type, random_seed=seed)
                except ValueError:
                    continue
                q.put(game_file_name)
            else:
                time.sleep(wait_time)
            counter += 1

    generator_processes = [mp.Process(target=data_generator_task, args=(p_num,)) for p_num in range(nb_worker)]

    for p in generator_processes:
        p.daemon = True
        p.start()

    return q


def game_generator(path="./", random_map=False, question_type="location", train_data_size=1):
    print("Generating %s games..." % str(train_data_size))
    res = set()

    if random_map:
        this_many_rooms = np.linspace(2, 12, train_data_size + 2)[1:-1]
        this_many_rooms = [int(item) for item in this_many_rooms]
        this_many_objects = []
        for i in range(len(this_many_rooms)):
            # ith game
            tmp = np.linspace(this_many_rooms[i] * 3, this_many_rooms[i] * 6, train_data_size + 2)[1:-1]
            tmp = [int(item) for item in tmp]
            if tmp[i] <= this_many_rooms[i] * 3:
                tmp[i] = this_many_rooms[i] * 3 + 1
            this_many_objects.append(tmp[i])
    else:
        this_many_rooms = 6
        this_many_objects = np.linspace(this_many_rooms * 3, this_many_rooms * 6, train_data_size + 2)[1:-1]
        this_many_objects = [int(item) for item in this_many_objects]
        for i in range(len(this_many_objects)):
            if this_many_objects[i] <= this_many_rooms * 3:
                this_many_objects[i] = this_many_rooms * 3 + 1

    while(True):
        if len(res) == train_data_size:
            break
        _id = len(res)
        try:
            if random_map:
                game_file_name = generate_random_map_games(len(res), path=path, question_type=question_type, random_seed=123 + _id, num_room=this_many_rooms[_id], num_object=this_many_objects[_id])
            else:
                game_file_name = generate_fixed_map_games(len(res), path=path, question_type=question_type, random_seed=123 + _id, num_object=this_many_objects[_id])
        except ValueError:
            continue
        res.add(game_file_name)
    print("Done generating games...")
    return list(res)
