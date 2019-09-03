import argparse
import datetime
import os
import copy
import time
import json
import visdom
import torch
import numpy as np
import tempfile
from os.path import join as pjoin
from distutils.dir_util import copy_tree

import gym
import textworld
from textworld.gym import register_game, make_batch2
from agent import Agent
import generic
import reward_helper
import game_generator
import evaluate
from query import process_facts

request_infos = textworld.EnvInfos(description=True,
                                   inventory=True,
                                   verbs=True,
                                   location_names=True,
                                   location_nouns=True,
                                   location_adjs=True,
                                   object_names=True,
                                   object_nouns=True,
                                   object_adjs=True,
                                   facts=True,
                                   last_action=True,
                                   game=True,
                                   admissible_commands=True,
                                   extras=["object_locations", "object_attributes", "uuid"])


def train(data_path):

    time_1 = datetime.datetime.now()
    agent = Agent()

    # visdom
    viz = visdom.Visdom()
    plt_win = None
    eval_plt_win = None
    viz_avg_correct_state_acc, viz_avg_qa_acc = [], []
    viz_eval_sufficient_info_reward, viz_eval_qa_reward = [], []

    step_in_total = 0
    running_avg_qa_reward = generic.HistoryScoreCache(capacity=500)
    running_avg_sufficient_info_reward = generic.HistoryScoreCache(capacity=500)
    running_avg_qa_loss = generic.HistoryScoreCache(capacity=500)
    running_avg_correct_state_loss = generic.HistoryScoreCache(capacity=500)

    output_dir, data_dir = ".", "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_sum_reward_so_far = 0.0
    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt")
            agent.update_target_net()
        elif os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()
        else:
            print("Failed to load pretrained model... couldn't find the checkpoint file...")

    # Create temporary folder for the generated games.
    games_dir = tempfile.TemporaryDirectory(prefix="tw_games")  # This is not deleted upon error. It would be better to use a with statement.
    games_dir = pjoin(games_dir.name, "")  # So path ends with '/'.
    # copy grammar files into tmp folder so that it works smoothly
    assert os.path.exists("./textworld_data"), "Oh no! textworld_data folder is not there..."
    os.mkdir(games_dir)
    os.mkdir(pjoin(games_dir, "textworld_data"))
    copy_tree("textworld_data", games_dir + "textworld_data")
    if agent.run_eval:
        assert os.path.exists(pjoin(data_path, agent.testset_path)), "Oh no! test_set folder is not there..."
        os.mkdir(pjoin(games_dir, agent.testset_path))
        copy_tree(pjoin(data_path, agent.testset_path), pjoin(games_dir, agent.testset_path))

    if agent.train_data_size == -1:
        game_queue_size = agent.batch_size * 5
        game_queue = []

    episode_no = 0
    if agent.train_data_size == -1:
        # endless mode
        game_generator_queue = game_generator.game_generator_queue(path=games_dir, random_map=agent.random_map, question_type=agent.question_type, max_q_size=agent.batch_size * 2, nb_worker=8)
    else:
        # generate the training set
        all_training_games = game_generator.game_generator(path=games_dir, random_map=agent.random_map, question_type=agent.question_type, train_data_size=agent.train_data_size)
        all_training_games.sort()
        all_env_ids = None
    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        if agent.train_data_size == -1:
            # endless mode
            for _ in range(agent.batch_size):
                if not game_generator_queue.empty():
                    tmp_game = game_generator_queue.get()
                    if os.path.exists(tmp_game):
                        game_queue.append(tmp_game)
            if len(game_queue) == 0:
                time.sleep(0.1)
                continue
            can_delete_these = []
            if len(game_queue) > game_queue_size:
                can_delete_these = game_queue[:-game_queue_size]
                game_queue = game_queue[-game_queue_size:]
            sampled_games = np.random.choice(game_queue, agent.batch_size).tolist()
            env_ids = [register_game(gamefile, request_infos=request_infos) for gamefile in sampled_games]
        else:
            if all_env_ids is None:
                all_env_ids = [register_game(gamefile, request_infos=request_infos) for gamefile in all_training_games]
            env_ids = np.random.choice(all_env_ids, agent.batch_size).tolist()

        if len(env_ids) != agent.batch_size:  # either less than or greater than
            env_ids = np.random.choice(env_ids, agent.batch_size).tolist()
        env_id = make_batch2(env_ids, parallel=True)
        env = gym.make(env_id)
        env.seed(episode_no)

        obs, infos = env.reset()
        batch_size = len(obs)
        # generate question-answer pairs here
        questions, answers, reward_helper_info = game_generator.generate_qa_pairs(infos, question_type=agent.question_type, seed=episode_no)
        print("====================================================================================", episode_no)
        print(questions[0], answers[0])

        agent.train()
        agent.init(obs, infos)

        commands, last_facts, init_facts = [], [], []
        commands_per_step, game_facts_cache = [], []
        for i in range(batch_size):
            commands.append("restart")
            last_facts.append(None)
            init_facts.append(None)
            game_facts_cache.append([])
            commands_per_step.append(["restart"])

        observation_strings, possible_words = agent.get_game_info_at_certain_step(obs, infos)
        observation_strings = [a + " <|> " + item for a, item in zip(commands, observation_strings)]
        input_quest, input_quest_char, _ = agent.get_agent_inputs(questions)

        transition_cache = []
        print_cmds = []
        counting_rewards_np = []
        valid_command_rewards_np = []

        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode
        # push init state into counting reward dict
        state_strings = agent.get_state_strings(infos)
        _ = agent.get_binarized_count(state_strings, update=True)
        for step_no in range(agent.max_nb_steps_per_episode):
            # update answerer input
            for i in range(batch_size):
                if agent.not_finished_yet[i] == 1:
                    agent.naozi.push_one(i, copy.copy(observation_strings[i]))
                if agent.prev_step_is_still_interacting[i] == 1:
                    new_facts = process_facts(last_facts[i], infos["game"][i], infos["facts"][i], infos["last_action"][i], commands[i])
                    game_facts_cache[i].append(new_facts)  # info used in reward computing of existence question
                    last_facts[i] = new_facts
                    if step_no == 0:
                        init_facts[i] = copy.copy(new_facts)

            # generate commands
            if agent.noisy_net:
                agent.reset_noise()  # Draw a new set of noisy weights

            observation_strings_w_history = agent.naozi.get()
            input_observation, input_observation_char, _ =  agent.get_agent_inputs(observation_strings_w_history)
            commands, replay_info = agent.act(obs, infos, input_observation, input_observation_char, input_quest, input_quest_char, possible_words, random=act_randomly)
            for i in range(batch_size):
                commands_per_step[i].append(commands[i])

            replay_info = [observation_strings_w_history, questions, possible_words] + replay_info
            admissible_commands = [set(item) - set(["look", "wait", "inventory"]) for item in infos["admissible_commands"]]
            vc_rewards = [float(c in ac) for c, ac in zip(commands, admissible_commands)]
            valid_command_rewards_np.append(np.array(vc_rewards))

            # pass commands into env
            obs, _, _, infos = env.step(commands)
            # possible words no not depend on history, because one can only interact with what is currently accessible
            observation_strings, possible_words = agent.get_game_info_at_certain_step(obs, infos)
            observation_strings = [a + " <|> " + item for a, item in zip(commands, observation_strings)]
            # counting rewards
            state_strings = agent.get_state_strings(infos)
            c_rewards = agent.get_binarized_count(state_strings, update=True)
            counting_rewards_np.append(np.array(c_rewards))

            if agent.noisy_net and step_in_total % agent.update_per_k_game_steps == 0:
                agent.reset_noise()  # Draw a new set of noisy weights

            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                interaction_loss = agent.update_interaction()
                if interaction_loss is not None:
                    running_avg_correct_state_loss.push(interaction_loss)
                qa_loss = agent.update_qa()
                if qa_loss is not None:
                    running_avg_qa_loss.push(qa_loss)

            print_cmds.append(commands[0] if agent.prev_step_is_still_interacting[0] else "--")
            # force stopping
            if step_no == agent.max_nb_steps_per_episode - 1:
                replay_info[-1] = torch.zeros_like(replay_info[-1])
            transition_cache.append(replay_info)
            step_in_total += 1
            if (step_no == agent.max_nb_steps_per_episode - 1 ) or (step_no > 0 and np.sum(generic.to_np(replay_info[-1])) == 0):
                break

        print(" / ".join(print_cmds))
        # The agent has exhausted all steps, now answer question.
        answerer_input = agent.naozi.get()
        answerer_input_observation, answerer_input_observation_char, answerer_observation_ids =  agent.get_agent_inputs(answerer_input)

        chosen_word_indices = agent.answer_question_act_greedy(answerer_input_observation, answerer_input_observation_char, answerer_observation_ids, input_quest, input_quest_char)  # batch
        chosen_word_indices_np = generic.to_np(chosen_word_indices)
        chosen_answers = [agent.word_vocab[item] for item in chosen_word_indices_np]
        # rewards
        # qa reward
        qa_reward_np = reward_helper.get_qa_reward(answers, chosen_answers)
        # sufficient info rewards
        masks = [item[-1] for item in transition_cache]
        masks_np = [generic.to_np(item) for item in masks]
        # 1 1 0 0 0 --> 1 1 0 0 0 0
        game_finishing_mask = np.stack(masks_np + [np.zeros((batch_size,))], 0)  # game step+1 x batch size
        # 1 1 0 0 0 0 --> 0 1 0 0 0
        game_finishing_mask = game_finishing_mask[:-1, :] - game_finishing_mask[1:, :]  # game step x batch size
        game_running_mask = np.stack(masks_np, 0)  # game step x batch size

        if agent.question_type == "location":
            # sufficient info reward: location question
            reward_helper_info["observation_before_finish"] = answerer_input
            reward_helper_info["game_finishing_mask"] = game_finishing_mask
            sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_location(reward_helper_info)
        elif agent.question_type == "existence":
            # sufficient info reward: existence question
            reward_helper_info["observation_before_finish"] = answerer_input
            reward_helper_info["game_facts_per_step"] = game_facts_cache  # facts before issuing command (we want to stop at correct state)
            reward_helper_info["init_game_facts"] = init_facts
            reward_helper_info["full_facts"] = infos["facts"]
            reward_helper_info["answers"] = answers
            reward_helper_info["game_finishing_mask"] = game_finishing_mask
            sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_existence(reward_helper_info)
        elif agent.question_type == "attribute":
            # sufficient info reward: attribute question
            reward_helper_info["answers"] = answers
            reward_helper_info["game_facts_per_step"] = game_facts_cache  # facts before and after issuing commands (we want to compare the differnce)
            reward_helper_info["init_game_facts"] = init_facts
            reward_helper_info["full_facts"] = infos["facts"]
            reward_helper_info["commands_per_step"] = commands_per_step  # commands before and after issuing commands (we want to compare the differnce)
            reward_helper_info["game_finishing_mask"] = game_finishing_mask
            sufficient_info_reward_np = reward_helper.get_sufficient_info_reward_attribute(reward_helper_info)
        else:
            raise NotImplementedError

        # push qa experience into qa replay buffer
        for b in range(batch_size):  # data points in batch
            # if the agent is not in the correct state, do not push it into replay buffer
            if np.sum(sufficient_info_reward_np[b]) == 0.0:
                continue
            agent.qa_replay_memory.push(False, qa_reward_np[b], answerer_input[b], questions[b], answers[b])

        # assign sufficient info reward and counting reward to the corresponding steps
        counting_rewards_np = np.stack(counting_rewards_np, 1)  # batch x game step
        valid_command_rewards_np = np.stack(valid_command_rewards_np, 1)  # batch x game step
        command_rewards_np = sufficient_info_reward_np + counting_rewards_np * game_running_mask.T * agent.revisit_counting_lambda + valid_command_rewards_np * game_running_mask.T * agent.valid_command_bonus_lambda  # batch x game step
        command_rewards = generic.to_pt(command_rewards_np, enable_cuda=agent.use_cuda, type="float")  # batch x game step
        for i in range(command_rewards_np.shape[1]):
            transition_cache[i].append(command_rewards[:, i])
        print(command_rewards_np[0])

        # push command generation experience into replay buffer
        for b in range(batch_size):
            is_prior = np.sum(command_rewards_np[b], 0) > 0.0
            for i in range(len(transition_cache)):
                batch_observation_strings, batch_question_strings, batch_possible_words, batch_chosen_indices, _, batch_rewards = transition_cache[i]
                is_final = True
                if masks_np[i][b] != 0:
                    is_final = False
                agent.command_generation_replay_memory.push(is_prior, batch_observation_strings[b], batch_question_strings[b], [item[b] for item in batch_possible_words], [item[b] for item in batch_chosen_indices], batch_rewards[b], is_final)
                if masks_np[i][b] == 0.0:
                    break

        # for printing
        r_qa = np.mean(qa_reward_np)
        r_sufficient_info = np.mean(np.sum(sufficient_info_reward_np, -1))
        running_avg_qa_reward.push(r_qa)
        running_avg_sufficient_info_reward.push(r_sufficient_info)
        print_rewards = np.mean(np.sum(command_rewards_np, -1))
        obs_string = answerer_input[0]
        print(obs_string)
        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        # close env
        env.close()
        if agent.train_data_size == -1:
            # when games are generated on the fly,
            # remove all files (including .json and .ni) that have been used
            files_to_delete = []
            for gamefile in can_delete_these:
                if not gamefile.endswith(".ulx"):
                    continue
                files_to_delete.append(gamefile)
                files_to_delete.append(gamefile.replace(".ulx", ".json"))
                files_to_delete.append(gamefile.replace(".ulx", ".ni"))
            # print("rm -f {}".format(" ".join(files_to_delete)))
            os.system("rm -f {}".format(" ".join(files_to_delete)))
        episode_no += batch_size

        time_2 = datetime.datetime.now()
        print("Episode: {:3d} | time spent: {:s} | interaction loss: {:2.3f} | qa loss: {:2.3f} | rewards: {:2.3f} | qa acc: {:2.3f}/{:2.3f} | correct state: {:2.3f}/{:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], running_avg_correct_state_loss.get_avg(), running_avg_qa_loss.get_avg(), print_rewards, r_qa, running_avg_qa_reward.get_avg(), r_sufficient_info, running_avg_sufficient_info_reward.get_avg()))

        if episode_no < agent.learn_start_from_this_episode:
            continue
        if episode_no == 0 or (episode_no % agent.save_frequency > (episode_no - batch_size) % agent.save_frequency):
            continue
        eval_qa_reward, eval_sufficient_info_reward = 0.0, 0.0
        # evaluate
        if agent.run_eval:
            eval_qa_reward, eval_sufficient_info_reward = evaluate.evaluate(data_dir, agent)
            # if run eval, then save model by eval accucacy
            if eval_qa_reward + eval_sufficient_info_reward > best_sum_reward_so_far:
                best_sum_reward_so_far = eval_qa_reward + eval_sufficient_info_reward
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
        # save model
        elif agent.save_checkpoint:
            if running_avg_qa_reward.get_avg() + running_avg_sufficient_info_reward.get_avg() > best_sum_reward_so_far:
                best_sum_reward_so_far = running_avg_qa_reward.get_avg() + running_avg_sufficient_info_reward.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")

        # plot using visdom
        viz_avg_correct_state_acc.append(running_avg_sufficient_info_reward.get_avg())
        viz_avg_qa_acc.append(running_avg_qa_reward.get_avg())
        viz_eval_sufficient_info_reward.append(eval_sufficient_info_reward)
        viz_eval_qa_reward.append(eval_qa_reward)
        viz_x = np.arange(len(viz_avg_correct_state_acc)).tolist()

        if plt_win is None:
            plt_win = viz.line(X=viz_x, Y=viz_avg_correct_state_acc,
                                opts=dict(title=agent.experiment_tag + "_train"),
                                name="correct state")
            viz.line(X=viz_x, Y=viz_avg_qa_acc,
                        opts=dict(title=agent.experiment_tag + "_train"),
                        win=plt_win, update='append', name="qa")
        else:
            viz.line(X=[len(viz_avg_correct_state_acc) - 1], Y=[viz_avg_correct_state_acc[-1]],
                        opts=dict(title=agent.experiment_tag + "_train"),
                        win=plt_win,
                        update='append', name="correct state")
            viz.line(X=[len(viz_avg_qa_acc) - 1], Y=[viz_avg_qa_acc[-1]],
                        opts=dict(title=agent.experiment_tag + "_train"),
                        win=plt_win,
                        update='append', name="qa")

        if eval_plt_win is None:
            eval_plt_win = viz.line(X=viz_x, Y=viz_eval_sufficient_info_reward,
                                    opts=dict(title=agent.experiment_tag + "_eval"),
                                    name="correct state")
            viz.line(X=viz_x, Y=viz_eval_qa_reward,
                        opts=dict(title=agent.experiment_tag + "_eval"),
                        win=eval_plt_win, update='append', name="qa")
        else:
            viz.line(X=[len(viz_eval_sufficient_info_reward) - 1], Y=[viz_eval_sufficient_info_reward[-1]],
                        opts=dict(title=agent.experiment_tag + "_eval"),
                        win=eval_plt_win,
                        update='append', name="correct state")
            viz.line(X=[len(viz_eval_qa_reward) - 1], Y=[viz_eval_qa_reward[-1]],
                        opts=dict(title=agent.experiment_tag + "_eval"),
                        win=eval_plt_win,
                        update='append', name="qa")

        # write accucacies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "sufficient info": running_avg_sufficient_info_reward.get_avg(),
                         "qa": running_avg_qa_reward.get_avg(),
                         "eval sufficient info": eval_sufficient_info_reward,
                         "eval qa": eval_qa_reward})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("data_path",
                        default="./",
                        help="where the data (games) are.")
    args = parser.parse_args()
    train(args.data_path)
