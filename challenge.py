# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


"""
.. _interactive_QA:

Interactive QA
==============

TODO

Settings
--------

    * nb_rooms (int): Number of rooms.
    * nb_entities (int): Number of entities (rooms + fixed in place + portable).
    * seed_map (int): Fix the random seed for the map generation.
    * with_placeholders (bool): Add as many placeholders as need to cover all possible attributes.

References
----------
TODO

"""

import re
import itertools
import argparse
import textwrap
from pprint import pprint
from collections import defaultdict, Counter

from typing import Mapping, Union, Dict, Optional, List

import numpy as np
import networkx as nx
from numpy.random import RandomState

import textworld
from textworld import GameMaker
from textworld.generator.maker import WorldRoom
from textworld.generator.world import World
from textworld.generator.game import Quest, Event, GameOptions
from textworld.generator.text_grammar import Grammar
from textworld.generator.graph_networks import DIRECTIONS, reverse_direction

from textworld.utils import encode_seeds
from textworld.utils import uniquify

from textworld.challenges.utils import get_seeds_for_game_generation
from textworld.challenges import register

RELEVANT_ATTRIBUTES = sorted(["edible", "drinkable", "portable", "openable",
                             "cuttable", "sharp", "heat_source", "cookable",
                             "holder"])


ROOM_NAMES = ["kitchen", "pantry", "livingroom", "bathroom", "bedroom",
              "backyard", "garden", "shed", "driveway", "street",
              "corridor", "supermarket"]

FOODS_COMPACT = {
    "egg" : {
        "properties": ["inedible", "cookable", "needs_cooking", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "milk" : {
        "indefinite": "some",
        "properties": ["drinkable", "inedible", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "water" : {
        "names": ["water", "orange juice", "tomato juice", "grape juice"],
        "indefinite": "some",
        "properties": ["drinkable", "inedible", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "bottle of water" : {
        "names": ["bottle of water", "bottle of sparkling water"],
        "properties": ["drinkable", "inedible", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "juice" : {
        "names": ["juice", "orange juice", "tomato juice", "grape juice"],
        "indefinite": "some",
        "properties": ["drinkable", "inedible", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "cooking oil" : {
        "names": ["vegetable oil", "peanut oil", "olive oil"],
        "indefinite": "some",
        "properties": ["inedible", "portable"],
        "locations": ["pantry.shelf", "supermarket.showcase"],
    },
    "chicken wing" : {
        "properties": ["inedible", "cookable", "needs_cooking", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "chicken leg" : {
        "properties": ["inedible", "cookable", "needs_cooking", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "chicken breast" : {
        "properties": ["inedible", "cookable", "needs_cooking", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "pork chop" : {
        "properties": ["inedible", "cookable", "needs_cooking", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "tuna" : {
        "names": ["red tuna", "white tuna"],
        "properties": ["inedible", "cookable", "needs_cooking", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "carrot" : {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "garden"],
    },
    "onion" : {
        "names": ["red onion", "white onion", "yellow onion"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "garden"],
    },
    "lettuce" : {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "garden"],
    },
    "potato" : {
        "names": ["red potato", "yellow potato", "purple potato"],
        "properties": ["inedible", "cookable", "needs_cooking", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.counter", "garden"],
    },
    "apple" : {
        "names": ["red apple", "yellow apple", "green apple"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.counter", "garden"],
    },
    "pineapple" : {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.counter", "garden"],
    },
    "banana" : {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.counter", "garden"],
    },
    "tomato" : {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.counter", "garden"],
    },
    "hot pepper" : {
        "names": ["red hot pepper", "green hot pepper"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.counter", "garden"],
    },
    "bell pepper" : {
        "names": ["red bell pepper", "yellow bell pepper", "green bell pepper", "orange bell pepper"],
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "garden"],
    },
    "black pepper" : {
        "properties": ["edible", "portable"],
        "locations": ["pantry.shelf", "supermarket.showcase"],
    },
    "flour" : {
        "properties": ["edible", "portable"],
        "locations": ["pantry.shelf", "supermarket.showcase"],
    },
    "salt" : {
        "properties": ["edible", "portable"],
        "locations": ["pantry.shelf", "supermarket.showcase"],
    },
    "sugar" : {
        "properties": ["edible", "portable"],
        "locations": ["pantry.shelf", "supermarket.showcase"],
    },
    "block of cheese" : {
        "properties": ["edible", "cookable", "raw", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "supermarket.showcase"],
    },
    "cilantro" : {
        "properties": ["edible", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "garden"],
    },
    "parsley" : {
        "properties": ["edible", "cuttable", "uncut", "portable"],
        "locations": ["kitchen.fridge", "garden"],
    }
}

FOODS = {}
for k, v in FOODS_COMPACT.items():
    if "names" in v:
        for name in v["names"]:
            FOODS[name] = dict(v)
            del FOODS[name]["names"]
    else:
        FOODS[k] = v


ENTITIES_COMPACT = {
    "cookbook": {
        "type": "o",
        "names": ["cookbook", "recipe book", "magazine"],
        "adjs": ["interesting"],
        "locations": ["kitchen.counter", "kitchen.table"],
        "properties": ["portable"],
        "desc": [None],
    },

    # Kitchen
    "fridge": {
        "type": "c",
        "names": ["fridge", "refrigerator"],
        "adjs": ["conventional", "stainless", "black", "modern"],
        "locations": ["kitchen"],
        "properties": ["openable", "holder", "fixed", "closeable", "lockable", "unlockable"],
        "desc": [None],
    },
    "counter": {
        "type": "s",
        "names": ["counter", "countertop"],
        "adjs": ["vast", "stainless", "marble"],
        "locations": ["kitchen"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "table": {
        "type": "s",
        "names": ["table", "kitchen island"],
        "adjs": ["massive", "stone", "wooden"],
        "locations": ["kitchen"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "stove": {
        "type": "stove",
        "names": ["stove", "cooktop"],
        "adjs": ["conventional", "stainless", "black"],
        "locations": ["kitchen"],
        "properties": ["holder", "heat_source", "fixed"],
        "desc": ["You can cook food with this [noun] to fry them."],
    },
    "oven": {
        "type": "oven",
        "names": ["oven"],
        "adjs": ["conventional", "stainless", "white", "black"],
        "locations": ["kitchen"],
        "properties": ["holder", "heat_source", "fixed", "openable", "closeable"],
        "desc": ["You can cook food with this [noun] to roast them."],
    },
    "glass": {
        "type": "o",
        "names": ["glass"],
        "adjs": ["wine", "white wine", "red wine", "juice", "cocktail"],
        "locations": ["kitchen.counter", "kitchen.table"],
        "properties": ["portable"],
        "desc": [None],
    },
    "mug": {
        "type": "o",
        "names": ["cup", "mug"],
        "adjs": ["coffee"],
        "locations": ["kitchen.counter", "kitchen.table"],
        "properties": ["portable"],
        "desc": [None],
    },
    "plate": {
        "type": "o",
        "names": ["plate"],
        "adjs": ["ceramic", "plastic", "dinner", "salad", "bread"],
        "locations": ["kitchen.counter", "kitchen.table"],
        "properties": ["portable"],
        "desc": [None],
    },
    "bowl": {
        "type": "o",
        "names": ["plate"],
        "adjs": ["ceramic", "plastic", "soup"],
        "locations": ["kitchen.counter", "kitchen.table"],
        "properties": ["portable"],
        "desc": [None],
    },
    "knife": {
        "type": "o",
        "names": ["knife"],
        "adjs": ["butter", "cooking", "pocket"],
        "locations": ["kitchen.counter", "kitchen.table"],
        "properties": ["sharp", "portable"],
        "desc": [None],
    },

    # Pantry
    "shelf": {
        "type": "s",
        "names": ["shelf"],
        "adjs": ["wooden"],
        "locations": ["pantry"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "stool": {
        "type": "s",
        "names": ["stool"],
        "adjs": ["wooden", "aluminium"],
        "locations": ["pantry"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },

    # Backyard
    "BBQ": {
        "type": "bbq",
        "names": ["bbq", "barbecue", "barbeque", "grill"],
        "adjs": ["old", "gas", "electric"],
        "locations": ["backyard"],
        "properties": ["heat_source", "fixed"],
        "desc": ["You can cook food with this [noun] to grill them."],
    },
    "patio table": {
        "type": "s",
        "names": ["patio table"],
        "adjs": ["stylish", "plastic"],
        "locations": ["backyard"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "patio chair": {
        "type": "s",
        "names": ["patio chair"],
        "adjs": ["stylish", "plastic"],
        "locations": ["backyard"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },

    # Supermarket
    "showcase": {
        "type": "s",
        "names": ["showcase"],
        "adjs": ["metallic", "refrigerated"],
        "locations": ["supermarket"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },

    # Livingroom
    "sofa": {
        "type": "s",
        "names": ["sofa", "couch"],
        "adjs": ["comfy", "sectional"],
        "locations": ["livingroom"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "remote": {
        "type": "o",
        "names": ["remote"],
        "adjs": ["tv", "cable", "universal"],
        "locations": ["livingroom"],
        "properties": ["portable"],
        "desc": [None],
    },
    "television": {
        "type": "o",
        "names": ["television", "tv"],
        "adjs": ["flat screen", "led"],
        "locations": ["livingroom"],
        "properties": ["portable"],
        "desc": [None],
    },

    # Bedroom
    "bed": {
        "type": "s",
        "names": ["bed"],
        "adjs": ["large", "king", "queen", "twin"],
        "locations": ["bedroom"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "dresser": {
        "type": "s",
        "names": ["dresser"],
        "adjs": ["large"],
        "locations": ["bedroom"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },

    # Bathroom
    "toilet": {
        "type": "s",
        "names": ["toilet"],
        "adjs": ["white"],
        "locations": ["bathroom"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "soap": {
        "type": "o",
        "names": ["soap"],
        "adjs": ["flower", "natural", "handmade"],
        "locations": ["bathroom"],
        "properties": ["portable"],
        "desc": [None],
    },
    "towel": {
        "type": "o",
        "names": ["towel"],
        "adjs": ["small", "large", "medium"],
        "locations": ["bathroom"],
        "properties": ["portable"],
        "desc": [None],
    },

    # Shed
    "workbench": {
        "type": "s",
        "names": ["workbench", "benchtop"],
        "adjs": ["wooden", "stainless", "stone"],
        "locations": ["shed"],
        "properties": ["holder", "fixed"],
        "desc": [None],
    },
    "toolbox": {
        "type": "c",
        "names": ["toolbox"],
        "adjs": ["metallic", "red"],
        "locations": ["shed"],
        "properties": ["openable", "holder", "fixed", "closeable", "lockable", "unlockable"],
        "desc": [None],
    },
    "tool": {
        "type": "o",
        "names": ["wrench", "screwdriver", "hammer"],
        "adjs": ["old", "new", "rusty"],
        "locations": ["shed"],
        "properties": ["portable"],
        "desc": [None],
    },
    # Misc
    "key": {
        "type": "k",
        "names": ["key"],
        "adjs": ["metal", "copper", "rusty", "red", "blue"],
        "locations": [""],
        "properties": ["portable"],
        "desc": [None],
    },

    # Door
    "door": {
        "type": "d",
        "names": ["door"],
        "adjs": ["sliding", "patio", "iron", "screen", "barn", "metallic", "aluminium", "red", "wooden", "fiberglass",
                 "glass", "front"],
        "locations": [""],
        "properties": ["openable", "closeable", "lockable", "unlockable", "fixed"],
        "desc": [None],
    },
    "gate": {
        "type": "d",
        "names": ["gate"],
        "adjs": ["iron", "metallic", "plastic", "front"],
        "locations": [""],
        "properties": ["openable", "closeable", "lockable", "unlockable", "fixed"],
        "desc": [None],
    },

}

ENTITIES = {}
for k in sorted(ENTITIES_COMPACT):
    v = ENTITIES_COMPACT[k]
    if "names" in v:
        for noun in v["names"]:
            for adj in v["adjs"]:
                if adj:
                    name = adj + " " + noun
                else:
                    name = noun

                ENTITIES[name] = dict(v)
                ENTITIES[name]["noun"] = noun
                ENTITIES[name]["adj"] = adj
                del ENTITIES[name]["names"]
                del ENTITIES[name]["adjs"]
    else:
        ENTITIES[k] = v

# Add food to entities list.
for k, v in FOODS.items():
    ENTITIES[k] = v
    ENTITIES[k]["type"] = "f"
    ENTITIES[k]["adj"] = ENTITIES[k].get("adj", "")
    ENTITIES[k]["noun"] = ENTITIES[k].get("noun", k)
    ENTITIES[k]["desc"] = ENTITIES[k].get("desc", [None])

ENTITIES_PER_TYPE = defaultdict(list)
for k in sorted(ENTITIES):
    v = ENTITIES[k]
    ENTITIES_PER_TYPE[v["type"]].append(k)


def _add_entity_by_name(M, entity_name, rng, holders):
    if entity_name in M.used_names:
        return None

    entity = M.new(type=ENTITIES[entity_name]["type"])
    entity.infos.name = entity_name
    entity.infos.adj = ENTITIES[entity.name]["adj"]
    entity.infos.noun = ENTITIES[entity.name]["noun"]
    entity.infos.desc = rng.choice(ENTITIES[entity.name]["desc"])
    entity.infos.indefinite = ENTITIES[entity.name].get("indefinite")

    for property_ in ENTITIES[entity.name]["properties"]:
        entity.add_property(property_)

    holders = list(holders)
    rng.shuffle(holders)
    for holder in holders:
        if _can_be_added(M, entity, holder):
            holder.add(entity)
            M.used_names.add(entity.name)
            return entity

    # Couldn't find a holder for the entity. Discard it.
    del M._entities[entity.id]
    return None


def _add_entity_by_noun(M, entity_noun, rng, holders):
    candidates = [k for k in sorted(ENTITIES.keys())
                  if ENTITIES[k]["noun"] == entity_noun and k not in M.used_names]
    if len(candidates) == 0:
        return None

    name = rng.choice(candidates)
    return _add_entity_by_name(M, name, rng, holders)


def _add_entity_by_attribute(M, attribute, rng, rooms, holders):
    candidates = [k for k in sorted(ENTITIES.keys())
                  if attribute in ENTITIES[k]["properties"] and k not in M.used_names and ENTITIES[k]["type"] != "d"]
    if len(candidates) == 0:
        return None

    name = rng.choice(candidates)
    if "fixed" in ENTITIES[name]["properties"]:
        holders = rooms

    return _add_entity_by_name(M, name, rng, holders)


def _create_entity(M, type, rng):
    candidates = [name for name in ENTITIES_PER_TYPE[type] if name not in M.used_names]

    if len(candidates) == 0:
        return None

    entity = M.new(type=type)
    entity.infos.name = entity.name or rng.choice(candidates)
    entity.infos.adj = ENTITIES[entity.name]["adj"]
    entity.infos.noun = ENTITIES[entity.name]["noun"]
    entity.infos.desc = rng.choice(ENTITIES[entity.name]["desc"])
    entity.infos.indefinite = ENTITIES[entity.name].get("indefinite")

    for property_ in ENTITIES[entity.name]["properties"]:
        entity.add_property(property_)

    M.used_names.add(entity.name)
    return entity


HEATING_SOURCES = set(["oven", "bbq", "stove"])

def _can_be_added(M, holder, room):
    CONTAINER_TYPES = [M.kb.logic.types.get("c")]
    CONTAINER_TYPES += list(itertools.chain(*M.kb.logic.types.multi_descendants(CONTAINER_TYPES)))
    CONTAINER_TYPES = set([t.name for t in CONTAINER_TYPES])
    SUPPORTER_TYPES = [M.kb.logic.types.get("s")]
    SUPPORTER_TYPES += list(itertools.chain(*M.kb.logic.types.multi_descendants(SUPPORTER_TYPES)))
    SUPPORTER_TYPES = set([t.name for t in SUPPORTER_TYPES])

    content_types = [entity.type for entity in room.content]

    if holder.type in CONTAINER_TYPES and len(CONTAINER_TYPES & set(content_types)) > 0:
        return False  # Room already has a container.

    if holder.type in SUPPORTER_TYPES and len(SUPPORTER_TYPES & set(content_types)) > 0:
        return False  # Room already has a supporter.

    if holder.type in HEATING_SOURCES and len(HEATING_SOURCES & set(content_types)) > 0:
        return False  # Room already has a heating source.

    return True


def make_game(settings: Mapping[str, str], options: Optional[GameOptions] = None) -> textworld.Game:
    """ Make a game for Interatice QA.

    Arguments:
        settings:
            Settings controlling the generation process (see notes).
        options:
            For customizing the game generation (see
            :py:class:`textworld.GameOptions <textworld.generator.game.GameOptions>`
            for the list of available options).

    Returns:
        Generated game.

    Notes:
        The settings are:

        * nb_rooms : Number of rooms.
        * nb_entities : Number of entities (rooms + fixed in place + portable).
        * seed_map : Fix the random seed for the map generation.
        * with_placeholders (bool): Add as many placeholders as need to cover all possible attributes.

    """
    options = options or GameOptions()
    options.grammar.allowed_variables_numbering = True  # TODO: DEBUG
    options.nb_rooms = settings["nb_rooms"]
    options.nb_objects = settings["nb_entities"]

    if settings.get("seed_map"):
        # Fixed the rooms layout and their name.
        options.seeds["map"] = settings["seed_map"]
        options.seeds["grammar"] = settings["seed_map"]

    rngs = options.rngs
    rng_map = rngs['map']
    rng_objects = rngs['objects']
    rng_grammar = rngs['grammar']
    rng_quest = rngs["quest"]

    grammar = Grammar(options.grammar, rng=rng_grammar)
    M = textworld.GameMaker(kb=options._kb, grammar=grammar)
    M.used_names = set()  # Going to use it to avoid duplicate names.

    # -= Map =-
    G = textworld.generator.make_map(n_rooms=options.nb_rooms, rng=rng_map,
                                     possible_door_states=None)
    assert len(G.nodes()) == options.nb_rooms
    rooms = M.import_graph(G)

    # Add doors
    paths = list(M.paths)
    rng_map.shuffle(paths)
    for path in paths[::2]:
        path.door = _create_entity(M, "d", rng_map)
        if rng_objects.rand() <= 0.25:
            M.add_fact("closed", path.door)
        else:
            M.add_fact("open", path.door)


    # Add properties to door
    for entity in M._entities.values():
        if entity.type == "d":
            M.add_fact("openable", entity)
            M.add_fact("closeable", entity)
            M.add_fact("lockable", entity)
            M.add_fact("unlockable", entity)

    # Assign name to rooms.
    room_names = list(ROOM_NAMES)
    rng_map.shuffle(room_names)
    for i, room in enumerate(rooms):
        room.infos.name = room_names[i]
        room.infos.noun = room_names[i]

    start_room = rng_quest.choice(rooms)
    M.set_player(start_room)

    # -= Objects =-
    RATIO_HOLDERS_MEAN = 0.3
    ratio_holders = rng_objects.normal(RATIO_HOLDERS_MEAN, 0.05)
    nb_holders = min(int(np.round(ratio_holders * options.nb_objects)), 2*len(rooms))
    nb_objects = options.nb_objects - len(rooms) - nb_holders
    assert nb_holders >= 0
    assert nb_objects >= 0

    # Add holder entities.
    HOLDER_TYPES = [M.kb.logic.types.get("c"), M.kb.logic.types.get("s")]
    HOLDER_TYPES += sorted(itertools.chain(*M.kb.logic.types.multi_descendants(HOLDER_TYPES)))
    HOLDER_TYPES = uniquify(HOLDER_TYPES)
    # Make sure we have enough holder names.
    NB_HOLDERS = sum(len(ENTITIES_PER_TYPE[t.name]) for t in HOLDER_TYPES)
    # print("Holders: {}/{}".format(nb_holders, NB_HOLDERS))
    assert nb_holders <= NB_HOLDERS, "Not enough holder variations."

    holders = []
    while len(holders) < nb_holders:
        # Create a new holder.
        holder_type = rng_objects.choice(HOLDER_TYPES)
        holder = _create_entity(M, type=holder_type.name, rng=rng_objects)
        if holder is None:
            continue

        # Place the holder in a room.
        shuffled_rooms = list(rooms)
        rng_objects.shuffle(shuffled_rooms)
        for room in shuffled_rooms:
            if _can_be_added(M, holder, room):
                room.add(holder)
                holders.append(holder)
                break

        # If closeable, close it.
        if M.kb.logic.types.get("c").is_supertype_of(holder_type):
            if rng_objects.rand() <= 0.25:
                M.add_fact("closed", holder)
            else:
                M.add_fact("open", holder)


    # Add the remaining entities.
    OBJECTS_TYPES = sorted(M.kb.logic.types.get("o").subtypes)

    # Make sure we have enough object names.
    NB_OBJECTS = sum(len(ENTITIES_PER_TYPE[t.name]) for t in OBJECTS_TYPES)
    # print("Objects: {}/{}".format(nb_objects, NB_OBJECTS))
    assert nb_objects <= NB_OBJECTS, "Not enough object variations."

    holders += rooms + [M.inventory]
    objects = []
    while len(objects) < nb_objects:
        # Create a new object.
        object_type = rng_objects.choice(OBJECTS_TYPES)
        obj = _create_entity(M, type=object_type.name, rng=rng_objects)
        if obj is None:
            continue

        objects.append(obj)

        # Add the object to an holder.
        holder = rng_objects.choice(holders)
        holder.add(obj)

    # Replace some entity names with placeholders (i.e. fake words).
    with_placeholders = settings["with_placeholders"]
    if with_placeholders:
        assert len(M.rooms) >= 2, "--with-placeholders needs at least two rooms."
        # Make sure we have at least two objects for each attribute.
        entities = [v for v in M._entities.values() if v.infos.type not in ["I", "P", "r", "d"]]
        attributes_covered = Counter(attr for e in entities for attr in ENTITIES[e.infos.name]["properties"])
        missing_attributes = [(attr, 2 - attributes_covered[attr]) for attr in RELEVANT_ATTRIBUTES if attributes_covered[attr] < 2]

        MAX_RETRY = 10
        cpt = 0
        while len(missing_attributes) > 0 and cpt < MAX_RETRY:
            rng_objects.shuffle(missing_attributes)
            obj = _add_entity_by_attribute(M, missing_attributes[0][0], rng_objects, M.rooms, holders)
            if obj is None:
                cpt += 1
                continue

            cpt = 0
            if obj and "holder" in ENTITIES[obj.infos.name]["properties"]:
                holders.append(obj)

                # If closeable, close it.
                if "closeable" in ENTITIES[obj.infos.name]["properties"]:
                    if rng_objects.rand() <= 0.25:
                        M.add_fact("closed", obj)
                    else:
                        M.add_fact("open", obj)

            entities = [v for v in M._entities.values() if v.infos.type not in ["I", "P", "r", "d"]]
            attributes_covered = Counter(attr for e in entities for attr in ENTITIES[e.infos.name]["properties"])
            missing_attributes = [(attr, 2 - attributes_covered[attr]) for attr in RELEVANT_ATTRIBUTES if attributes_covered[attr] < 2]

        if settings["verbose"]:
            if len(missing_attributes) > 0:
                print("MISSING:", missing_attributes)
                print("Try increasing MAX_RETRY")

        entities = [M._entities[k] for k in sorted(M._entities) if M._entities[k].infos.type not in ["I", "P", "r", "d"]]

        placeholder_objs = []
        for entity in entities:
            attributes_covered = set(attr for e in placeholder_objs for attr in ENTITIES[e.infos.name]["properties"])
            missing_attributes = set(RELEVANT_ATTRIBUTES) - attributes_covered

            remaining_attributes = set(attr for e in entities for attr in ENTITIES[e.infos.name]["properties"] if e not in placeholder_objs + [entity])
            if len(set(RELEVANT_ATTRIBUTES) - set(remaining_attributes)) > 0:
                continue

            if len(set(ENTITIES[entity.infos.name]["properties"]) & missing_attributes) > 0:
                placeholder_objs.append(entity)

        attributes_covered = Counter(attr for e in entities for attr in ENTITIES[e.infos.name]["properties"] if e not in placeholder_objs)
        if len(set(RELEVANT_ATTRIBUTES) - set(attributes_covered.keys())) > 0:
            print(attributes_covered)
            assert False, "Contact Marc is this happens!"

        # Load fake words generated with `fictionary -c 100 --max-length 10 > fake_words.txt`
        with open("vocabularies/fake_words.txt") as f:
            FAKE_WORDS = f.read().split()

        if settings["verbose"]:
            print("{:03d}/{:03d} entities anonymized.".format(len(placeholder_objs), len(M._entities)))

        rng_quest.shuffle(placeholder_objs)
        fake_words = list(FAKE_WORDS)  # Make a copy.
        rng_quest.shuffle(fake_words)
        for i, entity in enumerate(placeholder_objs):
            new_name = fake_words[i]
            if settings["verbose"]:
                print("{} => {}".format(entity.name, new_name))

            entity.infos.name = new_name
            entity.infos.noun = new_name
            entity.infos.adj = None
            entity.infos.desc = None  # Force regeneration of the descriptions.

    game = M.build()

    # Collect infos about this game.
    metadata = {
        "seeds": options.seeds,
        "settings": settings,
    }

    def _get_fqn(obj):
        """ Get fully qualified name """
        obj = obj.parent
        name = ""
        while obj:
            obj_name = obj.name
            if obj_name is None:
                obj_name = "inventory"

            name = obj_name + "." + name
            obj = obj.parent

        return name.rstrip(".")

    object_locations = {}
    object_attributes = {}
    for name in game.object_names:
        entity = M.find_by_name(name)
        if entity:
            if entity.type != "d":
                object_locations[name] = _get_fqn(entity)

            object_attributes[name] = [fact.name for fact in entity.properties]

    game.extras["object_locations"] = object_locations
    game.extras["object_attributes"] = object_attributes

    game.metadata = metadata
    uuid = "tw-interactive_qa-{specs}-{seeds}"
    uuid = uuid.format(specs=encode_seeds((options.nb_rooms, options.nb_objects)),
                       seeds=encode_seeds([options.seeds[k] for k in sorted(options.seeds)]))
    game.metadata["uuid"] = uuid
    game.extras["uuid"] = uuid
    return game


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()

    group = parser.add_argument_group('First TextWorld Competition game settings')
    group.add_argument("--recipe", type=int, default=1, metavar="INT",
                       help="Number of ingredients in the recipe. Default: %(default)s")
    group.add_argument("--take", type=int, default=0,  metavar="INT",
                       help="Number of ingredients to find. It must be less or equal to"
                            " the value of `--recipe`. Default: %(default)s")
    group.add_argument("--nb-rooms", type=int, default=1,
                       help="Number of locations in the game. Default: %(default)s")
    group.add_argument("--nb-entities", type=int, default=1,
                       help="Number of entities (rooms + fixed in place + portable). Default: %(default)s")
    group.add_argument("--seed-map", type=int,
                       help="Fixing the seed for the map generation. Default: random")
    group.add_argument("--with-placeholders", action="store_true",
                       help=" Add as many placeholders as need to cover all possible attributes.")
    return parser


register(name="tw-iqa",
         desc="Generate games for the Interactive QA dataset",
         make=make_game,
         add_arguments=build_argparser)
