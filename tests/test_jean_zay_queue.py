import unittest
from unittest.mock import patch

import main
from Management.connectToCluster import Servers
from Management.multiServerJob import queueJobs


class JeanZayQueueTests(unittest.TestCase):
    def test_size_scaling_plan_has_expected_settings(self):
        configs = main.jeanZaySizeScalingPlan(nrSeeds=2)

        self.assertEqual(len(configs), 10)
        self.assertEqual({config.rows for config in configs}, {50, 100, 150, 200, 250})
        self.assertEqual(
            {config.rows: config.nrThreads for config in configs},
            {50: 1, 100: 2, 150: 3, 200: 4, 250: 4},
        )
        self.assertTrue(
            all(
                config.reconnectionMethod == "edgeFlip"
                and config.reconnectRevert == 1
                and config.reconnectEdgeLocking == 0
                for config in configs
            )
        )

    @patch("main.queueJobs")
    def test_campaign_targets_jean_zay_with_standard_qos(self, queue_jobs):
        main.queueJeanZaySizeScalingJobs(submit=True, jobCopies=3, nrSeeds=1)

        args, kwargs = queue_jobs.call_args
        self.assertEqual(args[0], Servers.jeanZay)
        self.assertEqual(len(args[1]), 5)
        self.assertEqual(kwargs["jobCopies"], 3)
        self.assertEqual(kwargs["time_limit"], "20:00:00")
        self.assertEqual(kwargs["account"], "bph@cpu")
        self.assertEqual(kwargs["qos"], "qos_cpu-t3")
        self.assertTrue(kwargs["resume"])

    @patch("main.queueJobs")
    def test_smoke_test_is_small_and_uses_development_qos(self, queue_jobs):
        config = main.queueJeanZaySmokeTest(submit=True)

        self.assertEqual((config.rows, config.cols), (10, 10))
        self.assertEqual(config.maxLoad, 0.2)
        self.assertEqual(config.reconnectionMethod, "edgeFlip")
        self.assertEqual(config.reconnectRevert, 1)
        self.assertEqual(config.reconnectEdgeLocking, 0)
        args, kwargs = queue_jobs.call_args
        self.assertEqual(args[0], Servers.jeanZay)
        self.assertEqual(kwargs["qos"], "qos_cpu-dev")
        self.assertEqual(kwargs["time_limit"], "00:30:00")
        self.assertFalse(kwargs["resume"])

    @patch("main.queueJobs")
    def test_resume_smoke_uses_ten_short_singleton_copies(self, queue_jobs):
        config = main.queueJeanZayResumeSmokeTest(submit=True)

        self.assertEqual((config.rows, config.cols), (20, 20))
        self.assertEqual(config.maxLoad, 0.3)
        self.assertEqual(config.reconnectionMethod, "edgeFlip")
        self.assertEqual(config.reconnectRevert, 1)
        self.assertEqual(config.reconnectEdgeLocking, 0)
        args, kwargs = queue_jobs.call_args
        self.assertEqual(args[0], Servers.jeanZay)
        self.assertEqual(kwargs["jobCopies"], 10)
        self.assertEqual(kwargs["time_limit"], "00:01:00")
        self.assertEqual(kwargs["qos"], "qos_cpu-dev")
        self.assertTrue(kwargs["resume"])

    @patch("Management.multiServerJob.run_remote_command")
    def test_queue_payload_quotes_slurm_profile_values(self, run_remote_command):
        config = main.queueJeanZaySmokeTest(submit=False)

        queueJobs(
            Servers.jeanZay,
            [config],
            build=False,
            resume=False,
            time_limit="00:30:00",
            account="bph@cpu",
            qos="qos_cpu-dev",
        )

        remote_command = run_remote_command.call_args.args[1]
        self.assertIn("'\"time_limit\"': \"'00:30:00'\"", remote_command)
        self.assertIn("'\"account\"': \"'bph@cpu'\"", remote_command)
        self.assertIn("'\"qos\"': \"'qos_cpu-dev'\"", remote_command)


if __name__ == "__main__":
    unittest.main()
