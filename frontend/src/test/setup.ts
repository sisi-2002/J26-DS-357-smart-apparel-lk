import '@testing-library/jest-dom';

// jsdom does not implement ResizeObserver, and has no real layout engine
// (contentRect would always be 0 anyway). BboxCanvas (component1) uses it
// to measure the rendered image width for bbox coordinate scaling, so this
// stub fires the callback synchronously on observe() with a fixed width —
// letting tests assert the exact resulting scale deterministically.
const STUB_CONTAINER_WIDTH = 900;

if (globalThis.ResizeObserver === undefined) {
  class ResizeObserverStub implements ResizeObserver {
    readonly #callback: ResizeObserverCallback;
    constructor(callback: ResizeObserverCallback) {
      this.#callback = callback;
    }
    observe(target: Element) {
      const entry = {
        target,
        contentRect: { width: STUB_CONTAINER_WIDTH, height: 0 },
      } as unknown as ResizeObserverEntry;
      this.#callback([entry], this);
    }
    unobserve() {
      /* no-op stub */
    }
    disconnect() {
      /* no-op stub */
    }
  }
  globalThis.ResizeObserver = ResizeObserverStub;
}

export const RESIZE_OBSERVER_STUB_WIDTH = STUB_CONTAINER_WIDTH;
