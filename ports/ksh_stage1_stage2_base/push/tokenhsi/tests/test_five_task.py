"""Regression: preserve four tasks and add an independent PUSH_AT end to end."""
from pathlib import Path
import unittest
import torch
import yaml

from utils.edge_scenario_spec import (compose_canonical_graph, classify_templates, validate_graph,
                                     sample_graph, paired_placement_success)
from utils.task_role_spec import task_graph_packet, task_edges, TASKS
from utils.edge_context_spec import PUSH_CONTACT, AT
from utils.edge_ontop_spec import permute_graph
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata
from utils.unified_training import validate_unified_env, append_family, amp_family_ids, sample_family_matched, preserve_amp_labels
from utils.five_task import demo_distribution, size_probabilities
from learning.multi_agent.task_role_encoder import TaskTypeEmbeddingBias, task_action_condition

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return yaml.safe_load((ROOT/'data/cfg/multi_agent'/f'{name}.yaml').read_text())['env']


class FiveTaskTests(unittest.TestCase):
    def test_existing_packets_unchanged_and_fifth_id(self):
        graph = compose_canonical_graph(torch.tensor([[1, 2, 3, 4, 5]]))
        validate_graph(graph)
        packet = task_graph_packet(graph, 1).reshape(1, 5, 5)
        self.assertEqual(packet[0, :, 1].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(TASKS, ('sit','climb','carry_at','carry_ontop','push_at'))
        self.assertEqual(classify_templates(graph, torch.zeros(5, dtype=torch.long), torch.arange(5)).tolist(), [1,2,3,4,5])
        for k in range(4):
            old = compose_canonical_graph(torch.tensor([[k+1]]))
            torch.testing.assert_close(packet[:, k, 1:2], task_graph_packet(old, 1).reshape(1,5)[:, 1:2])
        permuted = permute_graph(graph, torch.rand(1,10).argsort(-1))
        torch.testing.assert_close(task_graph_packet(graph, 1), task_graph_packet(permuted, 1))

    def test_push_graph_has_contact_and_at_with_native_pair_saturation(self):
        graph = compose_canonical_graph(torch.tensor([[3,5]]))
        contact = graph.edge_relation == PUSH_CONTACT
        at = graph.edge_relation == AT
        self.assertEqual(int(contact.sum()), 1)
        own = torch.zeros_like(graph.edge_valid)
        own[at & (graph.edge_owner == 1)] = True
        saturated = paired_placement_success(own, graph)
        self.assertTrue(saturated[contact].all())
        self.assertFalse(saturated[(graph.edge_owner == 0)].any())
        valid, src, dst, tasks = task_edges(task_graph_packet(graph, 1))
        self.assertEqual(int(((tasks == 4)&valid).sum()), 3)

    def test_config_and_mixed_sampling(self):
        env = config('stage1_v2_fixed_five_task')
        validate_unified_env(env)
        graph = sample_graph(4096, env['relationGraph'])
        packet = task_graph_packet(graph, 4096).reshape(4096,2,5)
        counts = torch.bincount(packet[...,1].long().flatten(), minlength=5).float()/8192
        torch.testing.assert_close(counts, torch.tensor([.08,.2,.26,.26,.2]), atol=.025, rtol=0)
        fixed = sample_graph(2, env['relationGraph'], preset='push_at')
        self.assertTrue((task_graph_packet(fixed,2).reshape(2,2,5)[...,1] == 4).all())
        sizes = torch.tensor([[[.4,.4,.4], [1.3,1.8,1.3]]])
        weights = size_probabilities(sizes, env)
        self.assertEqual(weights.shape[-1], 6)
        self.assertEqual(weights[0,1].tolist(), [0,0,0,0,0,1])
        self.assertEqual(float(weights[0,0,5]), 0.)

    def test_fifth_embedding_receives_gradient_and_four_ids_stay_distinct(self):
        embedding = TaskTypeEmbeddingBias(4,2,5)
        packet = task_graph_packet(compose_canonical_graph(torch.tensor([[3,5]])),1)
        _, vectors, _ = task_action_condition(packet, embedding.category_embed)
        self.assertFalse(torch.equal(vectors[:,0], vectors[:,1]))
        vectors[:,1].square().sum().backward()
        self.assertGreater(float(embedding.category_embed.weight.grad[4].abs().sum()),0)
        self.assertEqual(float(embedding.category_embed.weight.grad[:4].abs().sum()),0)

    def test_amp_four_families_are_preserved_and_matched(self):
        env = config('stage1_v2_fixed_five_task')
        _, experts = demo_distribution(env)
        self.assertEqual(experts[0][-1],0.)
        self.assertEqual(experts[3][1:8],[0.]*7)
        labels = torch.arange(4)[:,None].expand(-1,2)
        obs = append_family(torch.randn(4,2,129),labels,4).flatten(1)
        self.assertEqual(obs.shape[-1],266)
        torch.testing.assert_close(obs.reshape(4,2,-1)[:,0,-4:],torch.eye(4))
        normalized = preserve_amp_labels(obs, obs*10+3, 2,4)
        torch.testing.assert_close(amp_family_ids(normalized,2,4), torch.arange(4))
        torch.testing.assert_close(normalized.reshape(4,2,-1)[...,-4:], obs.reshape(4,2,-1)[...,-4:])
        matched = sample_family_matched(obs.flip(0), obs,2,num_families=4)
        torch.testing.assert_close(matched,obs)
        with self.assertRaises(ValueError):
            sample_family_matched(obs[:3],obs,2,num_families=4)

    def test_checkpoint_contract_preserves_legacy_and_rejects_wrong_taxonomy(self):
        original = checkpoint_metadata(config('stage1_v2_fixed')['relationReward'])
        new = checkpoint_metadata(config('stage1_v2_fixed_five_task')['relationReward'])
        self.assertEqual(len(original['relation_taxonomy']),4)
        self.assertEqual(new['relation_taxonomy'], {name:i for i,name in enumerate(TASKS)})
        self.assertEqual(new['amp_conditioning']['label_width'],4)
        check_checkpoint_metadata({'relation_metadata':new},new)
        previous=dict(new)
        previous.pop('amp_conditioning')
        with self.assertRaises(ValueError):
            check_checkpoint_metadata({'relation_metadata':previous},new)
        with self.assertRaises(ValueError):
            check_checkpoint_metadata({'relation_metadata':original},new)

    def test_stage2_couples_only_push_and_shares_physical_source_and_goal(self):
        from utils.five_task_stage2 import sample_graph
        from utils.edge_scenario_spec import agent_object_indices, agent_goal_indices
        env=config('stage2_shared_full_success_five_task')
        validate_unified_env(env)
        sizes=torch.full((4096,4,3),.4)
        probabilities=size_probabilities(sizes,env)
        graph,groups,shared,templates,source=sample_graph(probabilities,env['relationGraph'],env['sharedSupport'])
        self.assertTrue((templates[shared]==5).all())
        self.assertEqual(set(templates[~shared].tolist()),{1,2,3,4,5})
        rows=torch.arange(len(groups)).repeat_interleave(4)
        owners=torch.arange(4).repeat(len(groups))
        objects=agent_object_indices(graph,rows,owners).reshape(-1,4)
        goals=agent_goal_indices(graph,rows,owners).reshape(-1,4)
        for a in range(4):
            for b in range(4):
                joint=shared[:,a] & (groups[:,a]==groups[:,b])
                torch.testing.assert_close(objects[joint,a],objects[joint,b])
                torch.testing.assert_close(goals[joint,a],goals[joint,b])
        packet=task_graph_packet(graph,len(groups)).reshape(-1,4,5)
        self.assertTrue((packet[...,1][shared]==4).all())

    def test_five_task_policy_forward_and_stage2_transfer(self):
        from torch import nn
        from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
        from learning.multi_agent.shared_support_network import SharedSupportBuilder
        from learning.multi_agent.stage2_network_builder import transfer_stage1
        from stage1_test_utils import observations
        torch.set_num_threads(1)
        def network(name,cls,m,o,width):
            env=config(name)
            train=yaml.safe_load((ROOT/'data/cfg/train/rlg'/f'{name}.yaml').read_text())
            builder=cls()
            builder.load(train['params']['network'])
            amp_width=1330 if name.endswith('_five_task') else 1320
            return builder.build('amp',actions_num=32,input_shape=(width,),amp_input_shape=(amp_width,),
                value_size=1,num_agents=m,num_objects=o,observation_mode='clean_scene',
                scene_entity_sizes=[223,30,1],scene_kinematic_size=7,scene_arena_scale=24.,
                relation_reward_mode=env['relationReward']['mode'],relation_graph_spec=env['relationGraph'],device='cpu')
        first=network('stage1_v2_fixed_five_task',AMPMultiAgentBuilder,2,4,634)
        original=network('stage1_v2_fixed',AMPMultiAgentBuilder,2,4,634)
        def same_architecture_except_new_task(before, after):
            old, new = before.state_dict(), after.state_dict()
            self.assertEqual(set(old), set(new))
            changed = {key for key in old if old[key].shape != new[key].shape}
            self.assertEqual(changed, {'actor_encoder.edge_encoder.category_embed.weight',
                                       'critic_encoder.edge_encoder.category_embed.weight',
                                       '_disc_mlp.0.weight'})
            for key in changed-{'_disc_mlp.0.weight'}:
                self.assertEqual(tuple(old[key].shape), (4,64))
                self.assertEqual(tuple(new[key].shape), (5,64))
            self.assertEqual(tuple(old['_disc_mlp.0.weight'].shape), (1024,1320))
            self.assertEqual(tuple(new['_disc_mlp.0.weight'].shape), (1024,1330))
        same_architecture_except_new_task(original, first)
        obs=observations(compose_canonical_graph(torch.tensor([[3,5],[1,2],[4,5]])))
        self.assertEqual(first.actor_encoder.edge_encoder.category_embed.num_embeddings,5)
        self.assertTrue(torch.isfinite(first.eval_actor(obs)[0]).all())
        self.assertTrue(torch.isfinite(first.eval_critic(obs)).all())
        self.assertTrue(torch.isfinite(first.eval_disc(torch.randn(3,1330))).all())
        target=network('stage2_shared_full_success_five_task',SharedSupportBuilder,4,12,1420)
        original_stage2=network('stage2_shared_full_success',SharedSupportBuilder,4,12,1420)
        same_architecture_except_new_task(original_stage2, target)
        source_model,target_model=nn.Module(),nn.Module()
        source_model.a2c_network=first
        target_model.a2c_network=target
        env=config('stage1_v2_fixed_five_task')
        checkpoint=dict(model=source_model.state_dict(),relation_metadata=checkpoint_metadata(env['relationReward']),
                        relation_experiment_config={'env':env},epoch=1)
        report=transfer_stage1(target_model,checkpoint)
        self.assertTrue(report['copied'])
        self.assertFalse(any(p.requires_grad for p in target.actor_encoder.parameters()))
        graph=compose_canonical_graph(torch.tensor([[1,2,3,5]]),num_objects=12)
        kin=torch.randn(1,20,7)
        kin[...,3:]=torch.nn.functional.normalize(kin[...,3:],dim=-1)
        scene=torch.cat((torch.randn(1,1256),kin.flatten(1),task_graph_packet(graph,1),torch.full((1,4),-1.)),-1)
        self.assertEqual(scene.shape[1],1420)
        self.assertTrue(torch.isfinite(target.eval_actor(scene)[0]).all())
        self.assertTrue(torch.isfinite(target.eval_critic(scene)).all())
        self.assertTrue(torch.isfinite(target.eval_disc(torch.randn(1,1330))).all())


if __name__ == '__main__':
    unittest.main()
