//! A minimal blocking executor.
//!
//! wgpu's native API is async, but the CLI and the tests are synchronous. This
//! is the whole executor they need: poll, and park until woken. `pollster` does
//! exactly this in about the same number of lines and is not on the project's
//! allowed dependency list, so it lives here instead.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            // `park_timeout` rather than `park`: some wgpu backends make
            // progress only when the device is polled from this same thread, so
            // a future can be Pending without anything ever calling our waker.
            // A bare `park` would deadlock there; this degrades to a slow spin.
            Poll::Pending => std::thread::park_timeout(Duration::from_millis(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_an_immediately_ready_future() {
        assert_eq!(block_on(std::future::ready(7)), 7);
    }

    #[test]
    fn resolves_a_future_woken_from_another_thread() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let f = flag.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            f.store(true, Ordering::SeqCst);
        });
        let flag2 = flag.clone();
        let fut = std::future::poll_fn(move |_cx| {
            if flag2.load(Ordering::SeqCst) {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        });
        block_on(fut);
        assert!(flag.load(Ordering::SeqCst));
    }
}
