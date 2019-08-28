import numpy as np
from generic import pad_sequences
from textworld.logic import Proposition


# Helper functions to query state/facts.
def _check_attributes(attributes, entity, facts):
    attributes = set(attributes)
    filtered = (fact for fact in facts
                if (fact.name in attributes and
                    fact.arguments[0].name == entity))
    return len(list(filtered)) >= len(attributes)


def _check_attribute(attribute, entity, facts):
    return _check_attributes([attribute], entity, facts)


def _check_relation(relation, arg1, arg2, facts):
    filtered = (fact for fact in facts
                if (fact.name == relation and
                    (arg1 is None or fact.arguments[0].name == arg1) and
                    (arg2 is None or fact.arguments[1].name == arg2)))
    return len(list(filtered)) > 0


def _get_entity_locations(facts):
    locations = {}
    for fact in facts:
        if fact.name in ("at", "in", "on"):
            locations[fact.arguments[0].name] = fact.arguments[1].name

    return locations


def _player_sees_the_entity(entity, facts):
    locations = _get_entity_locations(facts)
    if entity not in locations:
        return False

    entity_location = locations[entity]
    if entity_location in locations:
        if (_check_attributes(("openable", "closed"), entity_location, facts)):
            return False

        entity_location = locations[entity_location]

    return locations["P"] == entity_location or entity_location == "I"


def _determine_cmd_args(command, seen_entities):
    args = [entity for entity in seen_entities if entity in command]
    return sorted(args, key=lambda e: command.index(e))


def get_attribute_reward(prev_facts, facts, command, attribute, entity, answer):
    seen_entities = set(name for fact in (facts | prev_facts) for name in fact.names)

    if (entity not in seen_entities or
        (not _player_sees_the_entity(entity, prev_facts) and not _player_sees_the_entity(entity, facts))
    ):
        return False  # The agent can't see the entity.

    if attribute == "holder":
        if answer:
            # At some point, the agent observed there was an object in/on the entity.
            return (_check_relation("on", None, entity, facts) or
                    _check_relation("in", None, entity, facts))
        else:
            # In this toy dataset, if the entity is not a holder, then it is portable.
            if command == "take {}".format(entity):
                return True

            return False

    elif attribute == "portable":
        if answer:
            # At some point, the agent took the entity.
            return _check_relation("in", entity, "I", facts)
        else:
            # The agent tried to take the entity.
            if command == "take {}".format(entity):
                return True

            return False

    elif attribute == "openable":
        if answer:
            # At some point, the agent opened/closed the entity.
            if command == "open {}".format(entity):
                return (_check_attribute("closed", entity, prev_facts) and
                        _check_attribute("open", entity, facts))
            elif command == "close {}".format(entity):
                return (_check_attribute("open", entity, prev_facts) and
                        _check_attribute("closed", entity, facts))

            return False
        else:
            # The agent tried to open/close the entity.
            if (command == "open {}".format(entity) or command == "close {}".format(entity)):
                return True

            return False

    elif attribute == "drinkable":
        if answer:
            # At some point, the agent drank the entity.
            return _check_attribute("consumed", entity, facts)
        else:
            if _check_attribute("portable", entity, facts):
                # At some point, the agent tried to drink the entity.
                return command == "drink {}".format(entity)
            else:
                # At some point, the agent tried to take the entity which is non-portable.
                return command == "take {}".format(entity)

    elif attribute == "edible":
        if answer:
            # At some point, the agent ate the entity.
            return _check_attribute("consumed", entity, facts)
        else:
            if _check_attribute("portable", entity, facts):
                # At some point, the agent tried to eat the entity.
                return command == "eat {}".format(entity)
            else:
                # At some point, the agent tried to take the entity which is non-portable.
                return command == "take {}".format(entity)

    elif attribute == "sharp":
        if answer:
            # At some point, the agent cut something while holding the entity.
            if not (command.startswith("dice") or command.startswith("slice") or command.startswith("chop")):
                return False

            if not _check_relation("in", entity, "I", facts):
                return False  # The agent should be holding the entity.

            # TODO: the agent shouldn't be holding another sharp object!

            args = _determine_cmd_args(command, seen_entities)
            if len(args) == 0:
                return False  # Invalid command

            if not _check_attribute("uncut", args[0], prev_facts):
                return False  # Entity mentioned in the command was already cut.

            if (_check_attribute("sliced", args[0], facts) or
                _check_attribute("chopped", args[0], facts) or
                _check_attribute("diced", args[0], facts)
            ):
                return True  # The cutting command worked.

            return False
        else:
            if _check_attribute("portable", entity, facts):
                # At some point, the agent tried to cut something while holding the entity.
                if not (command.startswith("dice") or command.startswith("slice") or command.startswith("chop")):
                    return False  # Not a cutting command.

                if not _check_relation("in", entity, "I", facts):
                    return False  # The agent should be holding the entity.

                # TODO: the agent shouldn't be holding another sharp object!

                args = _determine_cmd_args(command, seen_entities)
                if len(args) == 0:
                    return False  # Invalid command

                if not _check_attribute("cuttable", args[0], facts):
                    return False  # The entity mentioned in the command is not cuttable.

                if (_check_attribute("sliced", args[0], prev_facts) or
                    _check_attribute("chopped", args[0], prev_facts) or
                    _check_attribute("diced", args[0], prev_facts)
                ):
                    return False  # Entity mentioned in the command was already cut.

                return True

            else:
                # At some point, the agent tried to take the entity which is non-portable.
                return command == "take {}".format(entity)

    elif attribute == "heat_source":
        if answer:
            # At some point, the agent cook something while being in that same location as the entity.
            if not command.startswith("cook"):
                return False

            args = _determine_cmd_args(command, seen_entities)
            if len(args) == 0:
                return False  # Invalid command

            if (_check_attribute("cooked", args[0], prev_facts) or
                _check_attribute("burned", args[0], prev_facts)
            ):
                return False  # Entity mentioned in the command was already cooked/burned.

            if (_check_attribute("cooked", args[0], facts) or
                _check_attribute("burned", args[0], facts)
            ):
                return True  # The cook command worked.

            return False
        else:
            if _check_attribute("portable", entity, facts):
                # Heat sources are not portable.
                return command == "take {}".format(entity)
            else:
                # At some point, the agent cook something while being in that same location as the entity.
                if not command.startswith("cook"):
                    return False

                args = _determine_cmd_args(command, seen_entities)
                if len(args) == 0:
                    return False  # Invalid command

                if not _check_attribute("cookable", args[0], facts):
                    return False  # The entity mentioned in the command is not cookable.

                if (_check_attribute("cooked", args[0], prev_facts) or
                    _check_attribute("burned", args[0], prev_facts)
                ):
                    return False  # Entity mentioned in the command was already cooked/burned.

                return True

    elif attribute == "cookable":
        if answer:
            # At some point, the agent cooked the entity.
            if command != "cook {}".format(entity):
                return False

            if (_check_attribute("cooked", entity, prev_facts) or
                _check_attribute("burned", entity, prev_facts)
            ):
                return False  # Entity mentioned in the command was already cooked.

            if (_check_attribute("cooked", entity, facts) or
                _check_attribute("burned", entity, facts)
            ):
                return True  # The cook command worked.

            return False
        else:
            if _check_attribute("portable", entity, facts):
                # At some point, the agent tried to cook the entity.
                if command != "cook {}".format(entity):
                    return False

                if not _check_relation("in", entity, "I", facts):
                    return False  # Agent was not holding the entity,

                # Check if there is a heat source at the player's location.
                locations = _get_entity_locations(facts)
                for entity, loc in locations.items():
                    if entity == "P" or loc != locations["P"]:
                        continue

                    if _check_attribute("heat_source", entity, facts):
                        return True  # Found a heat source in the same location as the player.

                return False

            else:
                # At some point, the agent tried to take the entity which is non-portable.
                return command == "take {}".format(entity)

    elif attribute == "cuttable":
        if answer:
            # At some point, the agent cut the entity.
            if (command != "slice {}".format(entity) and
                command != "dice {}".format(entity) and
                command != "chop {}".format(entity)
            ):
                return False

            if not _check_attribute("uncut", entity, prev_facts):
                return False  # Entity mentioned in the command was already cut.

            if not _check_attribute("uncut", entity, facts):
                return True  # The cook command worked.

            return False
        else:
            if _check_attribute("portable", entity, facts):
                # At some point, the agent cut the entity.
                if (command != "slice {}".format(entity) and
                    command != "dice {}".format(entity) and
                    command != "chop {}".format(entity)
                ):
                    return False

                if not _check_relation("in", entity, "I", facts):
                    return False  # Agent was not holding the entity,

                # Check if agent is holding a sharp object.
                locations = _get_entity_locations(facts)
                for entity, loc in locations.items():
                    if loc != "I":
                        continue

                    if _check_attribute("sharp", entity, facts):
                        return True  # Agent is holding a sharp object.

                return False

            else:
                # At some point, the agent tried to take the entity which is non-portable.
                return command == "take {}".format(entity)

    else:
        raise NotImplementedError


def check_reasoning_path_reward_sequence(asked_entity, asked_attribute, sequence_of_facts, commands, answer):
    rewards = [0] * len(sequence_of_facts)
    for t in range(1, len(sequence_of_facts)):
        if get_attribute_reward(sequence_of_facts[t-1], sequence_of_facts[t], commands[t], asked_attribute, asked_entity, answer):
            rewards[t] = 1  # Reasoning path has been completed.
            break

    return rewards


def get_sufficient_info_reward_attribute(reward_helper_info):
    asked_entities = reward_helper_info["_entities"]
    asked_attributes = reward_helper_info["_attributes"]
    init_game_facts = reward_helper_info["init_game_facts"]
    full_facts = reward_helper_info["full_facts"]
    answers = reward_helper_info["answers"]
    game_facts_per_step = reward_helper_info["game_facts_per_step"]  # batch x game step+1
    commands_per_step = reward_helper_info["commands_per_step"]  # batch x game step+1
    game_finishing_mask = reward_helper_info["game_finishing_mask"]  # game step x batch size
    rewards = []
    coverage_rewards = []
    seen_entity_reward = []
    for i in range(len(asked_entities)):  # Iterate over batch
        reward = check_reasoning_path_reward_sequence(asked_entities[i], asked_attributes[i],
                                                      game_facts_per_step[i], commands_per_step[i], bool(int(answers[i])))                                        
        rewards.append(reward)

        # add coverage
        end_facts = set()  # world discovered so far = union of observing game facts of all steps
        for t in range(len(game_facts_per_step[i])):
            end_facts = end_facts | set(game_facts_per_step[i][t])
        coverage = exploration_coverage(full_facts[i], end_facts, init_game_facts[i])
        coverage_rewards.append(coverage)

        seen_entities = set(name for f in end_facts for name in f.names)
        seen_entity_reward.append(1.0 if asked_entities[i] in seen_entities else 0.0)

    res = pad_sequences(rewards, dtype="float32")  # batch x game step
    res = res * game_finishing_mask.T
    coverage_rewards = np.array(coverage_rewards)
    seen_entity_reward = np.array(seen_entity_reward)
    res = res + game_finishing_mask.T * np.expand_dims(coverage_rewards + seen_entity_reward, axis=-1) * 0.1
    return res  # batch x game step


def get_sufficient_info_reward_location(reward_helper_info):
    asked_entities = reward_helper_info["_entities"]
    answers = reward_helper_info["_answers"]
    observation_before_finish = reward_helper_info["observation_before_finish"]
    game_finishing_mask = reward_helper_info["game_finishing_mask"]  # game step x batch size
    res = []
    for ent, a, obs in zip(asked_entities, answers, observation_before_finish):
        obs = obs.split()
        flag = True
        for w in ent.split() + a.split():
            if w not in obs:
                res.append(0.0)
                flag = False
                break
        if flag:
            res.append(1.0)
    res =  np.array(res)
    res = res.reshape((1, res.shape[0])) * game_finishing_mask
    return res.T  # batch x game step


def exploration_coverage(full_facts, end_facts, init_facts):
    containers_in_this_game = set([v for p in full_facts for v in p.arguments if v.type == "c"])
    rooms_in_this_game = set([v for p in full_facts for v in p.arguments if v.type == "r"])
    
    opened_container = set([p for p in end_facts for v in p.arguments if v in containers_in_this_game and p.name == "open"])
    visited_rooms = set([p for p in end_facts for v in p.arguments if v in rooms_in_this_game and p.name == "at" and p.arguments[0].name == "P"])
    
    init_opened_container = set([p for p in init_facts for v in p.arguments if v in containers_in_this_game and p.name == "open"])
    init_visited_rooms = set([p for p in init_facts for v in p.arguments if v in rooms_in_this_game and p.name == "at" and p.arguments[0].name == "P"])
    
    needs_to_be_discovered = len(containers_in_this_game) + len(rooms_in_this_game) - len(init_opened_container) - len(init_visited_rooms)
    discovered = len(opened_container) + len(visited_rooms) - len(init_opened_container) - len(init_visited_rooms)
    if needs_to_be_discovered == 0:
        return 0.0
    coverage = float(discovered) / float(needs_to_be_discovered)
    return max(coverage, 0.0)


def get_sufficient_info_reward_existence(reward_helper_info):

    sufficient_info_reward = []
    answers = reward_helper_info["answers"]
    game_facts_per_step = reward_helper_info["game_facts_per_step"]  # batch x step num
    init_game_facts = reward_helper_info["init_game_facts"]
    full_facts = reward_helper_info["full_facts"]
    asked_entities = reward_helper_info["_entities"]
    observation_before_finish = reward_helper_info["observation_before_finish"]
    game_finishing_mask = reward_helper_info["game_finishing_mask"]  # game step x batch size

    for i in range(len(observation_before_finish)):
        if answers[i] == "1":
            # if something exists, agent knows it as soon as it sees it
            for ent, obs in zip(asked_entities, observation_before_finish):
                obs = obs.split()
                flag = True
                for w in ent.split():
                    if w not in obs:
                        sufficient_info_reward.append(0.0)
                        flag = False
                        break
                if flag:
                    sufficient_info_reward.append(1.0)
        elif answers[i] == "0":
            # the agent has to exhaust room and containers to get this reward
            end_facts = set()  # world discovered so far = union of observing game facts of all steps
            for t in range(len(game_facts_per_step[i])):
                end_facts = end_facts | set(game_facts_per_step[i][t])
            coverage = exploration_coverage(full_facts[i], end_facts, init_game_facts[i])
            sufficient_info_reward.append(coverage)
        else:
            raise NotImplementedError
    res = np.array(sufficient_info_reward)
    res = res.reshape((1, res.shape[0])) * game_finishing_mask
    return res.T  # batch x game step


def get_qa_reward(answers, chosen_words):
    qa_reward = [float(gt == pred) for gt, pred in zip(answers, chosen_words)]
    return np.array(qa_reward)
