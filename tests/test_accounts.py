import asyncio
import unittest
from unittest.mock import MagicMock

from danyapi.accounts import AccountPool, AccountPoolBusy, DeepSeekAccount
from danyapi.deepseek.client import DeepSeekClient


def make_acct(i):
    client = MagicMock(spec=DeepSeekClient)
    client.index = i
    acct = DeepSeekAccount(i, client)
    return acct


class TestAccountPool(unittest.TestCase):
    def test_no_accounts_raises(self):
        async def run():
            pool = AccountPool([])
            with self.assertRaises(RuntimeError):
                await pool.acquire(None)

        asyncio.run(run())

    def test_session_affinity(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            pool = AccountPool([a0, a1])
            pool.register(1, "sess-abc")
            acct, sid = await pool.acquire("sess-abc")
            self.assertIs(acct, a1)
            self.assertEqual(sid, "sess-abc")
            acct, sid = await pool.acquire("sess-zzz")
            self.assertIsNone(sid)

        asyncio.run(run())

    def test_broken_account_releases_session(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            a0.broken = True
            pool = AccountPool([a0, a1])
            pool.register(0, "sess-broken")
            acct, sid = await pool.acquire("sess-broken")
            self.assertIs(acct, a1)
            self.assertIsNone(sid)
            self.assertNotIn("sess-broken", pool._by_session)

        asyncio.run(run())

    def test_round_robin_frees_first(self):
        async def run():
            a0, a1, a2 = make_acct(0), make_acct(1), make_acct(2)
            pool = AccountPool([a0, a1, a2])
            await a1.sem.acquire()
            await asyncio.sleep(0)
            self.assertTrue(a1.sem.locked())
            acct, _ = await pool.acquire(None)
            self.assertIs(acct, a0)
            await a0.sem.acquire()
            await asyncio.sleep(0)
            acct, _ = await pool.acquire(None)
            self.assertIs(acct, a2)
            await a2.sem.acquire()
            await asyncio.sleep(0)
            acct, _ = await pool.acquire(None)
            self.assertIsNotNone(acct)
            a1.sem.release()
            a0.sem.release()
            a2.sem.release()

        asyncio.run(run())

    def test_acquire_waits_for_free_account(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            pool = AccountPool([a0, a1])
            await a0.sem.acquire()
            await a1.sem.acquire()

            async def acquire():
                return await pool.acquire(None, max_wait=5)

            task = asyncio.create_task(acquire())
            await asyncio.sleep(0.1)
            self.assertFalse(task.done())
            a1.sem.release()
            acct, sid = await asyncio.wait_for(task, timeout=2)
            self.assertIs(acct, a1)
            self.assertIsNone(sid)
            a0.sem.release()

        asyncio.run(run())

    def test_acquire_times_out_when_busy(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            pool = AccountPool([a0, a1])
            await a0.sem.acquire()
            await a1.sem.acquire()
            with self.assertRaises(AccountPoolBusy):
                await pool.acquire(None, max_wait=0.2)
            a0.sem.release()
            a1.sem.release()

        asyncio.run(run())

    def test_session_affinity_waits_for_its_account(self):
        async def run():
            a0, a1 = make_acct(0), make_acct(1)
            pool = AccountPool([a0, a1])
            pool.register(1, "sess-x")
            await a1.sem.acquire()

            async def acquire():
                return await pool.acquire("sess-x", max_wait=5)

            task = asyncio.create_task(acquire())
            await asyncio.sleep(0.1)
            self.assertFalse(task.done())
            a1.sem.release()
            acct, sid = await asyncio.wait_for(task, timeout=2)
            self.assertIs(acct, a1)
            self.assertEqual(sid, "sess-x")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
