package main

import (
	"context"
	"testing"
	"time"

	"github.com/containerd/containerd/v2/core/content"
	digest "github.com/opencontainers/go-digest"
)

// blockingWriter is a content.Writer whose Write never returns on its
// own, simulating exactly the observed real-world failure: a registry
// PUT (an io.Pipe.Write, in the real code) that goes quiet mid-upload
// and never completes or fails by itself.
type blockingWriter struct{ unblock chan struct{} }

func (blockingWriter) Digest() digest.Digest                                              { return "" }
func (blockingWriter) Commit(context.Context, int64, digest.Digest, ...content.Opt) error { return nil }
func (blockingWriter) Status() (content.Status, error)                                    { return content.Status{}, nil }
func (blockingWriter) Truncate(int64) error                                               { return nil }
func (blockingWriter) Close() error                                                       { return nil }
func (w blockingWriter) Write(p []byte) (int, error) {
	<-w.unblock // never sent in the "stall" test — simulates a write that never returns
	return len(p), nil
}

func TestStallWriterCancelsOnBlockedWrite(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	sw := newStallWriter(blockingWriter{unblock: make(chan struct{})}, 100*time.Millisecond, cancel)
	defer sw.stop()

	done := make(chan struct{})
	go func() {
		sw.Write([]byte("hello")) // blocks forever; only returns if the test itself unblocks it
		close(done)
	}()

	select {
	case <-ctx.Done():
		// expected: the stall timer fired and canceled ctx while Write
		// was still blocked.
	case <-done:
		t.Fatal("Write returned before it should have (test double is broken)")
	case <-time.After(2 * time.Second):
		t.Fatal("stall timer never canceled ctx despite Write blocking indefinitely")
	}
}

func TestStallWriterResetsOnProgress(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	unblock := make(chan struct{})
	sw := newStallWriter(blockingWriter{unblock: unblock}, 150*time.Millisecond, cancel)
	defer sw.stop()

	// Let the write succeed well within the stall timeout — ctx must
	// NOT be canceled just because a write happened, only if one never
	// returns.
	go func() {
		time.Sleep(20 * time.Millisecond)
		close(unblock)
	}()

	if _, err := sw.Write([]byte("hello")); err != nil {
		t.Fatalf("Write: %v", err)
	}

	select {
	case <-ctx.Done():
		t.Fatal("ctx was canceled despite Write completing well within the timeout")
	case <-time.After(300 * time.Millisecond):
		// expected: no cancellation.
	}
}
