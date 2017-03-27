#!/usr/bin/env python3
import abc
import asyncio
import locale
import re
import signal
import sys
from asyncio.subprocess import PIPE
from contextlib import closing

import numpy as np
import sounddevice as sd


class Generator:
    CHANNELS = 1
    RATE = 44700

    @abc.abstractmethod
    def reinit(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next(self, samples_num: int) -> np.ndarray:
        raise NotImplementedError()


class SimpleGenerator(Generator):
    def __init__(self):
        self._current = 0

    def reinit(self):
        self._current = 0

    def get_next(self, samples_num: int):
        time_range = self._get_time_range(samples_num)
        return self._generate(time_range)

    def _get_time_range(self, samples_num: int):
        samples_range = np.linspace(
            self._current, self._current + samples_num, samples_num,
            endpoint=False, dtype=np.float64)
        self._current += samples_num
        return np.divide(samples_range, self.RATE)

    def _generate(self, time_range: np.ndarray):
        return None


class ConstantGenerator(Generator):
    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def reinit(self):
        pass

    def get_next(self, samples_num: int):
        return np.repeat([self.value], samples_num)


def ensure_generator(value):
    if isinstance(value, Generator):
        return value
    else:
        return ConstantGenerator(value)


class GeneratorProperty:
    def __init__(self):
        self.value = None

    def __get__(self, instance, owner) -> np.ndarray:
        return self.value

    def __set__(self, instance, value: np.ndarray):
        self.value = ensure_generator(value)


class SineGenerator(SimpleGenerator):
    freq = GeneratorProperty()
    volume = GeneratorProperty()

    def __init__(self, freq, volume):
        super().__init__()
        self.freq = freq
        self.volume = volume

    def _generate(self, time_range: np.ndarray):
        freq = self.freq.get_next(time_range.size)
        vol = self.volume.get_next(time_range.size)
        return self.__sine(freq, time_range).astype(np.float16) * vol

    @staticmethod
    def __sine(freq: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.sin(freq * t * 2. * np.pi)


class ExpAttackGenerator(SimpleGenerator):
    duration = GeneratorProperty()
    mute = GeneratorProperty()

    def __init__(self, duration, mute: bool = False):
        super().__init__()
        self.duration = duration
        self.mute = mute
        self.x = 0

    def _generate(self, time_range: np.ndarray):
        dur = self.duration.get_next(time_range.size)
        attack = self.__exp_attack(dur, time_range).astype(np.float16)
        attack[(time_range > dur) | self.mute.get_next(time_range.size)] = 0
        return attack

    @staticmethod
    def __exp_attack(duration: np.ndarray, t: np.ndarray) -> np.ndarray:
        return 5 * t * np.exp(1 - 5 * t / duration) / duration


class PingSignalGenerator(Generator):
    def __init__(self):
        super().__init__()
        self._attack_g = ExpAttackGenerator(0.1, True)
        self._sine_g = SineGenerator(freq=1, volume=self._attack_g)

    def reinit(self):
        self._attack_g.mute = True
        self._attack_g.reinit()

    def get_next(self, samples_num: int):
        return self._sine_g.get_next(samples_num)

    def ping(self, latency):
        # todo: зависимость частоты от задержки
        self._sine_g.freq = 1500 - 300 * np.log(latency)
        self._attack_g.mute = False
        self._attack_g.reinit()


async def ping(host: str, callback: callable):
    def parse_ping_line(s: str):
        match = re.match(".*time=(\d*(\.\d+)?) .*", s)
        if not match:
            return None
        return float(match.group(1))

    process = await asyncio.create_subprocess_exec("ping", host, stdout=PIPE)
    async for line in process.stdout:
        latency = parse_ping_line(
            line.decode(locale.getpreferredencoding(False)))
        if latency:
            callback(latency)


async def play_noise(generator: Generator, duration: int = None,
                     block_size: int = 128):
    def callback(outdata, frames, time, status):
        data = generator.get_next(frames).reshape(frames, generator.CHANNELS)
        outdata[:] = data.reshape(frames, generator.CHANNELS)

    with sd.OutputStream(channels=generator.CHANNELS, callback=callback,
                         samplerate=generator.RATE, blocksize=block_size):
        if duration is None:
            while True:
                await asyncio.sleep(2)
        else:
            await asyncio.sleep(duration)


def run_loop_until_interrupt(loop):
    def cancel_all_tasks():
        for task in asyncio.Task.all_tasks():
            task.cancel()
        asyncio.ensure_future(stop_the_loop())

    async def stop_the_loop():
        asyncio.get_event_loop().stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, cancel_all_tasks)
    loop.run_forever()
    loop.close()


def cli():
    try:
        hostname = sys.argv[1]
    except IndexError:
        print("Usage: aping <hostname>", file=sys.stderr)
        sys.exit(2)

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.get_event_loop()

    with closing(loop):
        signal_generator = PingSignalGenerator()
        asyncio.ensure_future(play_noise(signal_generator))

        def tune_signal_generator(latency):
            print(latency)
            signal_generator.ping(latency)

        asyncio.ensure_future(ping(hostname, tune_signal_generator))
        run_loop_until_interrupt(loop)
