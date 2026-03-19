package main

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ellistarn/muse/internal/anthropic"
	"github.com/ellistarn/muse/internal/bedrock"
	"github.com/ellistarn/muse/internal/inference"
	museOpenAI "github.com/ellistarn/muse/internal/openai"
	"github.com/ellistarn/muse/internal/storage"
)

var bucket string
var verbose bool

func newRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "muse",
		Short: "The distilled essence of how you think",
		Long: `A muse absorbs your conversations from agent interactions, distills them into
muse.md, and embodies your unique thought processes when asked questions.

Workflow:

  1. muse distill    Discover conversations, observe, and distill muse.md
  2. muse show       Print muse.md
  3. muse ask        Ask your muse a question (stateless, one-shot)
  4. muse listen     Start an MCP server so agents can ask your muse

Getting started:

  muse distill && muse show

Data is stored locally at ~/.muse/ by default. Set MUSE_BUCKET to use S3 instead.
Set MUSE_PROVIDER=anthropic to use the Anthropic API directly (requires ANTHROPIC_API_KEY).
Default provider is bedrock.

Run "muse listen --help" for MCP server configuration.`,
		SilenceErrors: true,
		SilenceUsage:  true,
	}
	cmd.PersistentFlags().StringVar(&bucket, "bucket", os.Getenv("MUSE_BUCKET"), "S3 bucket name (or set MUSE_BUCKET)")
	cmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "show per-item progress during pipeline stages")
	cmd.AddCommand(newDistillCmd())
	cmd.AddCommand(newShowCmd())
	cmd.AddCommand(newListenCmd())
	cmd.AddCommand(newAskCmd())
	cmd.AddCommand(newSyncCmd())
	return cmd
}

// newStore returns an S3-backed store when a bucket is configured,
// otherwise a local filesystem store rooted at ~/.muse/.
func newStore(ctx context.Context) (storage.Store, error) {
	if bucket != "" {
		return storage.NewS3Store(ctx, bucket)
	}
	store, err := storage.NewLocalStore()
	if err != nil {
		return nil, err
	}
	return store, nil
}

// Model tiers used by the pipeline. Compose is for editorial work (final
// muse composition, ask). Observe handles bulk work (observation, labeling,
// summarization).
const (
	TierCompose = "compose"
	TierObserve = "observe"
)

// newLLMClient creates an inference.Client based on MUSE_PROVIDER and tier.
func newLLMClient(ctx context.Context, tier string) (inference.Client, error) {
	provider := os.Getenv("MUSE_PROVIDER")
	switch provider {
	case "anthropic":
		return anthropic.NewClient(anthropicModel(tier))
	case "openai":
		return museOpenAI.NewClient(openaiModel(tier))
	case "bedrock", "":
		return bedrock.NewClient(ctx, bedrockModel(tier))
	default:
		return nil, fmt.Errorf("unknown MUSE_PROVIDER %q (use 'anthropic', 'openai', or 'bedrock')", provider)
	}
}

func bedrockModel(tier string) string {
	if tier == TierObserve {
		return bedrock.ModelSonnet
	}
	return bedrock.ModelOpus
}

func anthropicModel(tier string) string {
	if tier == TierObserve {
		return anthropic.ModelSonnet
	}
	return anthropic.ModelOpus
}

func openaiModel(tier string) string {
	if tier == TierObserve {
		return museOpenAI.ModelMini
	}
	return museOpenAI.ModelReasoning
}
