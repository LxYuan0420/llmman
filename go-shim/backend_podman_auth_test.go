//go:build podman

package main

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPodmanLogoutDoesNotPanicOnSuccess guards the nil-Stdout crash.
// Logout, not Login, because only its success path is reachable offline:
// it needs just an auth file holding a credential, whereas Login's needs
// a real docker.CheckAuth round-trip. Same bug in both.
func TestPodmanLogoutDoesNotPanicOnSuccess(t *testing.T) {
	const registry = "registry.example.com"
	authFile := filepath.Join(t.TempDir(), "auth.json")

	body, err := json.Marshal(map[string]any{
		"auths": map[string]any{
			registry: map[string]string{
				"auth": base64.StdEncoding.EncodeToString([]byte("user:pass")),
			},
		},
	})
	if err != nil {
		t.Fatalf("marshal auth file: %v", err)
	}
	if err := os.WriteFile(authFile, body, 0o600); err != nil {
		t.Fatalf("write auth file: %v", err)
	}
	// The only way to aim podmanLogout, which builds its own empty
	// SystemContext, at a throwaway file rather than the real one.
	t.Setenv("REGISTRY_AUTH_FILE", authFile)

	// Before the fix this did not return: the test binary aborted with a
	// nil-pointer dereference inside auth.Logout.
	if err := podmanLogout(registry); err != nil {
		t.Fatalf("podmanLogout(%q): %v", registry, err)
	}

	after, err := os.ReadFile(authFile)
	if err != nil {
		t.Fatalf("read auth file back: %v", err)
	}
	if strings.Contains(string(after), registry) {
		t.Fatalf("credentials for %s still present after logout: %s", registry, after)
	}
}
