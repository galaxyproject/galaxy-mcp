// The full surface, for a Node host: the browser entry plus what needs a local filesystem or
// node:crypto (the biocontainer recommender hashes with it).
import "./operations/all";
export * from "./index.browser";

export { downloadDatasetOp, downloadDataset, type DownloadDatasetResult } from "./operations/download-dataset";
export { uploadFileOp, uploadFile, type UploadFileResult } from "./operations/upload-file";
export {
  recommendBiocontainerOp,
  recommendBiocontainer,
  type BiocontainerRecommendation,
} from "./operations/recommend-biocontainer";
export {
  recommendContainer,
  biocontainerTagBuilt,
  QUAY_BIOCONTAINERS_PREFIX,
  type ContainerRecommendation,
  type MatchQuality,
  type PackageSpec,
  type RecommendationSource,
} from "./mulled";
