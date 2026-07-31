// Package main is the CGO entrypoint for the llmman Go shim.
// It is compiled as a C static archive and linked into the Rust binary.
// Build tags select either the Docker (containerd) or Podman backend.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"os"
	"unsafe"

	"github.com/sirupsen/logrus"
)

// LLMMAN_DEBUG=1 (or any non-empty value) surfaces containerd's own
// request/response debug logging — including the exact registry scope
// requested and the token/auth challenge exchange — which is otherwise
// only logged at debug level and invisible by default. Useful for
// diagnosing registry-side auth/scope rejections (401/403/"insufficient_
// scope") without needing to instrument anything by hand.
func init() {
	if os.Getenv("LLMMAN_DEBUG") != "" {
		logrus.SetLevel(logrus.DebugLevel)
	}
}

// response is the JSON envelope returned by every exported function.
// Rust deserialises this to decide success/failure.
type response struct {
	OK    bool   `json:"ok"`
	Data  string `json:"data,omitempty"`
	Error string `json:"error,omitempty"`
}

func okResp(data string) *C.char {
	b, _ := json.Marshal(response{OK: true, Data: data})
	return C.CString(string(b))
}

func errResp(err error) *C.char {
	b, _ := json.Marshal(response{OK: false, Error: err.Error()})
	return C.CString(string(b))
}

func errMsg(msg string) *C.char {
	b, _ := json.Marshal(response{OK: false, Error: msg})
	return C.CString(string(b))
}

// llmman_free releases a C string previously returned by this library.
//
//export llmman_free
func llmman_free(s *C.char) {
	C.free(unsafe.Pointer(s))
}

// main is required for -buildmode=c-archive.
func main() {}
