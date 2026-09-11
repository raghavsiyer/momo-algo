// Virtual-clock playback over the loader's record store.
//
// Subscribers receive batched records per animation frame while playing, and
// a full rebuilt snapshot on seek. Each subscriber implements:
//   { onBatch(batch), onSeek(snapshot), }
// where batch  = {t, ticks[], books[], bars[], states[], fills[]}
// and snapshot = {t, book, state, ticksBefore, barsBefore, fillsBefore}.
import { lastIndexAtOrBefore } from './loader.js';

export const SPEEDS = [1, 2, 5, 10, 25];

export class Player {
  constructor(loader) {
    this.loader = loader;
    this.startMs = loader.meta.window.start_ms;
    this.endMs = loader.meta.window.end_ms;
    this.t = this.startMs;
    this.speed = 5;
    this.playing = false;
    this.buffering = false;
    this.finished = false;
    this._cursor = { ticks: 0, books: 0, bars: 0, states: 0, fills: 0 };
    this._subscribers = [];
    this._raf = null;
    this._lastFrame = null;
    this.pendingSeek = null; // target beyond loaded data; retried as chunks land
    this.onClock = null; // (t, {playing, buffering, finished, speed})
  }

  subscribe(subscriber) {
    this._subscribers.push(subscriber);
  }

  _emitClock() {
    if (this.onClock) {
      this.onClock(this.t, {
        playing: this.playing,
        buffering: this.buffering,
        finished: this.finished,
        speed: this.speed,
      });
    }
  }

  play() {
    if (this.finished) this.seek(this.startMs);
    this.playing = true;
    this._lastFrame = null;
    this._emitClock();
    if (this._raf === null) this._raf = requestAnimationFrame((ts) => this._frame(ts));
  }

  pause() {
    this.playing = false;
    if (this._raf !== null) { cancelAnimationFrame(this._raf); this._raf = null; }
    this._emitClock();
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  setSpeed(speed) {
    this.speed = speed;
    this._emitClock();
  }

  stepSeconds(seconds) {
    this.seek(Math.min(this.endMs, Math.max(this.startMs, this.t + seconds * 1000)));
  }

  _frame(frameTs) {
    this._raf = null;
    if (!this.playing) return;
    if (this._lastFrame === null) this._lastFrame = frameTs;
    const dt = Math.min(1000, frameTs - this._lastFrame);
    this._lastFrame = frameTs;

    const target = this.t + dt * this.speed;
    const loadedThrough = this.loader.loadedThroughMs();
    if (target > loadedThrough && !this.loader.buildDone()) {
      this.buffering = true;
      this._emitClock();
      this._raf = requestAnimationFrame((ts) => { this._lastFrame = ts; this._frame(ts); });
      return;
    }
    this.buffering = false;
    this.t = Math.min(target, this.endMs);
    this._drainTo(this.t);
    this._emitClock();
    if (this.t >= this.endMs) {
      this.finished = true;
      this.pause();
      return;
    }
    this._raf = requestAnimationFrame((ts) => this._frame(ts));
  }

  _drainTo(targetMs) {
    const batch = { t: targetMs, ticks: [], books: [], bars: [], states: [], fills: [] };
    for (const kind of ['ticks', 'books', 'bars', 'states', 'fills']) {
      const records = this.loader[kind];
      let cursor = this._cursor[kind];
      while (cursor < records.length && records[cursor].t <= targetMs) {
        batch[kind].push(records[cursor]);
        cursor += 1;
      }
      this._cursor[kind] = cursor;
    }
    if (batch.ticks.length || batch.books.length || batch.bars.length || batch.states.length || batch.fills.length) {
      for (const subscriber of this._subscribers) subscriber.onBatch(batch);
    }
  }

  seek(targetMs) {
    const clamped = Math.min(this.endMs, Math.max(this.startMs, targetMs));
    const loadedThrough = this.loader.loadedThroughMs();
    if (clamped > loadedThrough && !this.loader.buildDone()) {
      // Not loaded yet — go to the loaded edge now and finish the jump when
      // the target chunk arrives (checkPendingSeek fires from onProgress).
      this.pendingSeek = clamped;
      this.buffering = true;
      this.t = Math.max(this.startMs, loadedThrough);
    } else {
      this.pendingSeek = null;
      this.buffering = false;
      this.t = clamped;
    }
    this.finished = this.t >= this.endMs;

    const snapshot = this._snapshotAt(this.t);
    for (const kind of Object.keys(this._cursor)) {
      this._cursor[kind] = lastIndexAtOrBefore(this.loader[kind], this.t) + 1;
    }
    for (const subscriber of this._subscribers) subscriber.onSeek(snapshot);
    this._emitClock();
  }

  checkPendingSeek() {
    if (this.pendingSeek === null) return;
    if (this.loader.loadedThroughMs() >= this.pendingSeek || this.loader.buildDone()) {
      const target = this.pendingSeek;
      this.pendingSeek = null;
      this.seek(target);
    }
  }

  _snapshotAt(targetMs) {
    const loader = this.loader;
    const bookIdx = lastIndexAtOrBefore(loader.books, targetMs);
    const stateIdx = lastIndexAtOrBefore(loader.states, targetMs);
    const tickIdx = lastIndexAtOrBefore(loader.ticks, targetMs);
    const barIdx = lastIndexAtOrBefore(loader.bars, targetMs);
    const fillIdx = lastIndexAtOrBefore(loader.fills, targetMs);
    return {
      t: targetMs,
      book: bookIdx >= 0 ? loader.books[bookIdx] : null,
      state: stateIdx >= 0 ? loader.states[stateIdx] : null,
      lastTick: tickIdx >= 0 ? loader.ticks[tickIdx] : null,
      ticksBefore: tickIdx + 1,   // count of ticks with t <= target
      barsBefore: barIdx + 1,
      fillsBefore: fillIdx + 1,
    };
  }
}
