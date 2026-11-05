#!/usr/bin/env python3
import re
import shutil
import sys
from pathlib import Path


def read(path):
    return Path(path).read_text(encoding="utf-8")


def write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"Could not find patch target: {label}")
    return text.replace(old, new, 1)


def insert_before(text, marker, insertion, label):
    if insertion.strip() in text:
        return text
    if marker not in text:
        raise RuntimeError(f"Could not find insertion marker: {label}")
    return text.replace(marker, insertion + marker, 1)


def extract_between(text, start, end, label):
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"Could not find extraction start: {label}")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"Could not find extraction end: {label}")
    return text[i:j]


def patch_cmake(root):
    path = root / "CMakeLists.txt"
    text = read(path)
    if "src/SemanticMasker.cc" not in text:
        text = replace_once(text, "src/Settings.cc\n", "src/Settings.cc\nsrc/SemanticMasker.cc\n", "cmake source list")
    if "include/SemanticMasker.h" not in text:
        text = replace_once(text, "include/Settings.h\n", "include/Settings.h\ninclude/SemanticMasker.h\n", "cmake header list")
    write(path, text)


def patch_semantic_files(root, semantic):
    include_dir = root / "include"
    src_dir = root / "src"
    shutil.copy2(semantic / "include" / "SemanticMasker.h", include_dir / "SemanticMasker.h")
    shutil.copy2(semantic / "src" / "SemanticMasker.cc", src_dir / "SemanticMasker.cc")

    header = include_dir / "SemanticMasker.h"
    text = read(header)
    if "dyneomtCommand" not in text:
        text = replace_once(
            text,
            '    std::string tokenmaskDataset = "ade20k";\n',
            '    std::string tokenmaskDataset = "ade20k";\n'
            '    std::string dyneomtCommand;\n'
            '    std::string dyneomtWorkDir = "/tmp/dynslam_dyneomt";\n',
            "dyneomt string config",
        )
        text = replace_once(
            text,
            "    float tokenmaskOverlapThreshold = 0.8f;\n",
            "    float tokenmaskOverlapThreshold = 0.8f;\n"
            "    float dyneomtThreshold = 0.5f;\n"
            "    bool dyneomtOutputIsFeatureMask = false;\n",
            "dyneomt numeric config",
        )
    write(header, text)

    cc = src_dir / "SemanticMasker.cc"
    text = read(cc)
    if "#include <cstdio>" not in text:
        text = replace_once(text, "#include <cstdint>\n", "#include <cstdint>\n#include <cstdio>\n", "cstdio include")
    if "#include <cerrno>" not in text:
        text = replace_once(text, "#include <cctype>\n", "#include <cctype>\n#include <cerrno>\n", "cerrno include")

    if "ShellQuote" not in text:
        shell_quote = r'''
std::string ShellQuote(const std::string &value)
{
    std::string quoted = "'";
    for(char c : value)
    {
        if(c == '\'')
            quoted += "'\\''";
        else
            quoted += c;
    }
    quoted += "'";
    return quoted;
}

'''
        text = replace_once(text, "}\n\nstd::string Trim(const std::string &value)", "}\n\n" + shell_quote + "std::string Trim(const std::string &value)", "shell quote helper")

    if "class DynEOMTExternalBackend" not in text:
        dyn_backend = r'''
class DynEOMTExternalBackend : public ISemanticBackend
{
public:
    explicit DynEOMTExternalBackend(const SemanticMaskerConfig &config)
        : mConfig(config), mFrameCounter(0)
    {
        if(mConfig.dyneomtCommand.empty())
            throw std::runtime_error("SemanticMasker: dyneomt backend requires Semantic.dyneomt_command");
        if(mConfig.dyneomtWorkDir.empty())
            mConfig.dyneomtWorkDir = "/tmp/dynslam_dyneomt";
        EnsureDirectory(mConfig.dyneomtWorkDir);
    }

    void PredictFeatureMask(const cv::Mat &bgr, cv::Mat &featureMask) override
    {
        if(bgr.empty())
        {
            featureMask.release();
            return;
        }

        const unsigned long long frameId = ++mFrameCounter;
        const std::string stem = "frame_" + std::to_string(frameId);
        const std::string inputPath = JoinPath(mConfig.dyneomtWorkDir, stem + "_rgb.png");
        const std::string outputPath = JoinPath(mConfig.dyneomtWorkDir, stem + "_dynamic.png");

        if(!cv::imwrite(inputPath, bgr))
        {
            throw std::runtime_error("SemanticMasker: failed to write DynEOMT input " + inputPath);
        }
        std::remove(outputPath.c_str());

        std::ostringstream command;
        command << "DYN_SLAM_INPUT=" << ShellQuote(inputPath) << " "
                << "DYN_SLAM_OUTPUT=" << ShellQuote(outputPath) << " "
                << "DYN_SLAM_WIDTH=" << bgr.cols << " "
                << "DYN_SLAM_HEIGHT=" << bgr.rows << " "
                << mConfig.dyneomtCommand;

        const int code = std::system(command.str().c_str());
        if(code != 0)
        {
            std::remove(inputPath.c_str());
            throw std::runtime_error("SemanticMasker: DynEOMT command failed with code " + std::to_string(code));
        }

        cv::Mat mask = cv::imread(outputPath, cv::IMREAD_GRAYSCALE);
        std::remove(inputPath.c_str());
        std::remove(outputPath.c_str());
        if(mask.empty())
        {
            throw std::runtime_error("SemanticMasker: DynEOMT command did not produce " + outputPath);
        }

        if(mask.size() != bgr.size())
            cv::resize(mask, mask, bgr.size(), 0.0, 0.0, cv::INTER_NEAREST);

        cv::threshold(mask, mask, std::max(0.0f, std::min(1.0f, mConfig.dyneomtThreshold)) * 255.0f,
                      255.0, cv::THRESH_BINARY);

        if(mConfig.dyneomtOutputIsFeatureMask)
            featureMask = mask;
        else
            DynamicToFeatureMask(mask, mConfig.dilatePixels, featureMask);
    }

private:
    static void EnsureDirectory(const std::string &path)
    {
        if(path.empty())
            return;
        struct stat st;
        if(::stat(path.c_str(), &st) == 0)
        {
            if((st.st_mode & S_IFDIR) != 0)
                return;
            throw std::runtime_error("SemanticMasker: path exists but is not a directory: " + path);
        }
        if(::mkdir(path.c_str(), 0775) != 0 && errno != EEXIST)
            throw std::runtime_error("SemanticMasker: failed to create directory: " + path);
    }

    SemanticMaskerConfig mConfig;
    unsigned long long mFrameCounter;
};

'''
        text = replace_once(
            text,
            "\n#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS\nclass TrtLogger",
            "\n" + dyn_backend + "#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS\nclass TrtLogger",
            "dyneomt backend class",
        )

    if 'backend == "dyneomt"' not in text:
        text = replace_once(
            text,
            "std::unique_ptr<ISemanticBackend> CreateBackend(const SemanticMaskerConfig &config)\n{\n#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS\n",
            "std::unique_ptr<ISemanticBackend> CreateBackend(const SemanticMaskerConfig &config)\n{\n"
            "    const std::string backend = ToLower(config.backend);\n"
            "    if(backend == \"dyneomt\" || backend == \"dyneomt_external\")\n"
            "        return std::unique_ptr<ISemanticBackend>(new DynEOMTExternalBackend(config));\n"
            "#ifdef LITESEMSLAM_USE_TENSORRT_SEMANTICS\n",
            "create backend prelude",
        )
        text = replace_once(text, "    const std::string backend = ToLower(config.backend);\n    if(backend == \"segnet\")", "    if(backend == \"segnet\")", "duplicate backend local")
        text = replace_once(
            text,
            "#else\n    (void)config;\n    throw std::runtime_error(\"LiteSemSLAM was built without TensorRT semantic support\");\n#endif\n}",
            "#else\n"
            "    if(backend == \"segnet\" || backend == \"maskrcnn\" || backend == \"tokenmask\")\n"
            "        throw std::runtime_error(\"SemanticMasker: TensorRT semantic backend was not built in this binary\");\n"
            "    throw std::runtime_error(\"SemanticMasker: unsupported semantic backend: \" + config.backend);\n"
            "#endif\n}",
            "create backend fallback",
        )
    text = text.replace("SemanticMasker TensorRT backend ready", "SemanticMasker backend ready")
    write(cc, text)


def patch_frame(root, semantic):
    target_h = root / "include" / "Frame.h"
    target_cc = root / "src" / "Frame.cc"
    ant_h = read(semantic / "include" / "Frame.h")
    ant_cc = read(semantic / "src" / "Frame.cc")

    text = read(target_h)
    if "ApplySemanticMask" not in text:
        text = replace_once(
            text,
            "    void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);\n\n",
            "    void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);\n\n"
            "    int ApplySemanticMask(const cv::Mat &leftFeatureMask, const cv::Mat &rightFeatureMask = cv::Mat(),\n"
            "                          int minKeptFeatures = 0, float minKeepRatio = 0.0f);\n\n",
            "Frame.h ApplySemanticMask declaration",
        )
    if "mSemanticFeatureMask" not in text:
        text = replace_once(
            text,
            "    std::vector<cv::KeyPoint> mvKeysUn;\n\n",
            "    std::vector<cv::KeyPoint> mvKeysUn;\n\n"
            "    cv::Mat mSemanticFeatureMask, mSemanticFeatureMaskRight;\n"
            "    std::vector<cv::KeyPoint> mvSemanticRejectedKeys, mvSemanticRejectedKeysRight;\n\n",
            "Frame.h semantic state",
        )
    write(target_h, text)

    text = read(target_cc)
    if "mGrid[i][j].clear();" not in text:
        text = replace_once(
            text,
            "    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)\n"
            "        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){\n"
            "            mGrid[i][j].reserve(nReserve);\n"
            "            if(Nleft != -1){\n"
            "                mGridRight[i][j].reserve(nReserve);\n"
            "            }\n"
            "        }\n",
            "    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)\n"
            "        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){\n"
            "            mGrid[i][j].clear();\n"
            "            mGrid[i][j].reserve(nReserve);\n"
            "            mGridRight[i][j].clear();\n"
            "            if(Nleft != -1){\n"
            "                mGridRight[i][j].reserve(nReserve);\n"
            "            }\n"
            "        }\n",
            "Frame.cc grid clear",
        )
    helpers = extract_between(ant_cc, "namespace\n{\nbool MaskKeepsKeyPoint", "\n}\n\nFrame::Frame", "Frame.cc semantic helpers")
    if "MaskKeepsKeyPoint" not in text:
        text = replace_once(
            text,
            "cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);\n\n",
            "cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);\n\n" + helpers + "\n}\n\n",
            "Frame.cc helper insertion",
        )
    if "mSemanticFeatureMask" not in text:
        text = replace_once(
            text,
            "     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),\n",
            "     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),\n"
            "     mSemanticFeatureMask(frame.mSemanticFeatureMask.clone()), mSemanticFeatureMaskRight(frame.mSemanticFeatureMaskRight.clone()),\n"
            "     mvSemanticRejectedKeys(frame.mvSemanticRejectedKeys), mvSemanticRejectedKeysRight(frame.mvSemanticRejectedKeysRight),\n"
            "     mvuRight(frame.mvuRight),\n",
            "Frame.cc copy constructor semantic members",
        )
    if "int Frame::ApplySemanticMask" not in text:
        apply_mask = extract_between(ant_cc, "int Frame::ApplySemanticMask", "\n#ifdef ORB_SLAM3_USE_CUDA_FRONTEND", "Frame.cc ApplySemanticMask implementation")
        text = insert_before(text, "void Frame::SetPose", apply_mask + "\n\n", "Frame.cc SetPose")
    write(target_cc, text)


def patch_tracking_header(root):
    path = root / "include" / "Tracking.h"
    text = read(path)
    if '#include "SemanticMasker.h"' not in text:
        text = replace_once(text, '#include "MapDrawer.h"\n', '#include "MapDrawer.h"\n#include "SemanticMasker.h"\n', "Tracking.h SemanticMasker include")
    if "#include <memory>" not in text:
        text = replace_once(text, "#include <mutex>\n", "#include <mutex>\n#include <memory>\n", "Tracking.h memory include")
    if "ConfigureSemanticMasker" not in text:
        insertion = (
            "    void ConfigureSemanticMasker(const string &strSettingPath);\n"
            "    unsigned long long SubmitSemanticFrame(const cv::Mat &image, double timestamp);\n"
            "    bool ShouldUseSemanticMaskForCurrentFrame() const;\n"
            "    void ApplySemanticMaskForFrame(unsigned long long requestId);\n\n"
            "    SemanticMaskerConfig mSemanticConfig;\n"
            "    std::unique_ptr<SemanticMasker> mpSemanticMasker;\n\n"
        )
        text = replace_once(text, "    void newParameterLoader(Settings* settings);\n", insertion + "    void newParameterLoader(Settings* settings);\n", "Tracking.h semantic members")
    write(path, text)


def semantic_parser_block(semantic):
    ant = read(semantic / "src" / "Tracking.cc")
    body = extract_between(
        ant,
        "bool ReadOptionalSemanticInt",
        "\n}\n\n\nTracking::Tracking",
        "Tracking.cc semantic parser",
    )
    if "Semantic.dyneomt_command" not in body:
        body = body.replace(
            '    if(ReadOptionalSemanticString(fSettings, "Semantic.dynamic_labels", stringValue))\n'
            "        config.dynamicLabels = stringValue;\n",
            '    if(ReadOptionalSemanticString(fSettings, "Semantic.dynamic_labels", stringValue))\n'
            "        config.dynamicLabels = stringValue;\n"
            '    if(ReadOptionalSemanticString(fSettings, "Semantic.dyneomt_command", stringValue))\n'
            "        config.dyneomtCommand = stringValue;\n"
            '    if(ReadOptionalSemanticString(fSettings, "Semantic.dyneomt_workdir", stringValue))\n'
            "        config.dyneomtWorkDir = stringValue;\n"
            '    if(ReadOptionalSemanticString(fSettings, "Semantic.dyneomt_work_dir", stringValue))\n'
            "        config.dyneomtWorkDir = stringValue;\n",
        )
        body = body.replace(
            '    if(ReadOptionalSemanticInt(fSettings, "Semantic.input_is_rgb", intValue))\n'
            "        config.inputIsRgb = (intValue != 0);\n",
            '    if(ReadOptionalSemanticInt(fSettings, "Semantic.input_is_rgb", intValue))\n'
            "        config.inputIsRgb = (intValue != 0);\n"
            '    if(ReadOptionalSemanticInt(fSettings, "Semantic.dyneomt_output_is_feature_mask", intValue))\n'
            "        config.dyneomtOutputIsFeatureMask = (intValue != 0);\n",
        )
        body = body.replace(
            '    if(ReadOptionalSemanticFloat(fSettings, "Semantic.tokenmask_overlap_threshold", floatValue))\n'
            "        config.tokenmaskOverlapThreshold = floatValue;\n",
            '    if(ReadOptionalSemanticFloat(fSettings, "Semantic.tokenmask_overlap_threshold", floatValue))\n'
            "        config.tokenmaskOverlapThreshold = floatValue;\n"
            '    if(ReadOptionalSemanticFloat(fSettings, "Semantic.dyneomt_threshold", floatValue))\n'
            "        config.dyneomtThreshold = floatValue;\n",
        )
    return "namespace\n{\n" + body + "\n}\n\n"


def semantic_methods_block():
    return r'''
void Tracking::ConfigureSemanticMasker(const string &strSettingPath)
{
    try
    {
        mSemanticConfig = ParseOptionalSemanticMaskerConfig(strSettingPath);
        if(!mSemanticConfig.enabled)
            return;

        mpSemanticMasker.reset(new SemanticMasker(mSemanticConfig));
        if(!mpSemanticMasker->Start())
            mpSemanticMasker.reset();
    }
    catch(const std::exception &e)
    {
        std::cerr << "Tracking: semantic masker disabled: " << e.what() << std::endl;
        mpSemanticMasker.reset();
        mSemanticConfig.enabled = false;
    }
}

unsigned long long Tracking::SubmitSemanticFrame(const cv::Mat &image, double timestamp)
{
    if(!mpSemanticMasker || !mpSemanticMasker->IsEnabled())
        return 0;
    return mpSemanticMasker->Submit(image, timestamp);
}

bool Tracking::ShouldUseSemanticMaskForCurrentFrame() const
{
    if(!mpSemanticMasker || !mpSemanticMasker->IsEnabled())
        return false;
    if(mSemanticConfig.skipUntilInitialized && mState != OK)
        return false;
    return true;
}

void Tracking::ApplySemanticMaskForFrame(unsigned long long requestId)
{
    if(!requestId || !mpSemanticMasker || !mpSemanticMasker->IsEnabled())
        return;

    cv::Mat featureMask;
    if(!mpSemanticMasker->WaitForMask(requestId, featureMask) || featureMask.empty())
        return;

    const int removed = mCurrentFrame.ApplySemanticMask(featureMask, cv::Mat(),
                                                       mSemanticConfig.minKeptFeatures,
                                                       mSemanticConfig.minKeepRatio);
    if(removed > 0 && getenv("DYNSLAM_SEMANTIC_TRACE"))
        std::cout << "DynSLAM semantic mask removed " << removed << " features" << std::endl;
}

'''


def patch_tracking_cc(root, semantic):
    path = root / "src" / "Tracking.cc"
    text = read(path)
    if "#include <stdexcept>" not in text:
        if "#include <cstdlib> // For std::exit\n" in text:
            text = replace_once(text, "#include <cstdlib> // For std::exit\n", "#include <cstdlib> // For std::exit\n#include <stdexcept>\n", "Tracking.cc stdexcept include")
        else:
            text = replace_once(text, "#include <csignal>\n", "#include <csignal>\n#include <stdexcept>\n", "Tracking.cc stdexcept include")
    if "ParseOptionalSemanticMaskerConfig" not in text:
        text = insert_before(text, "void Tracking::signalHandler", semantic_parser_block(semantic), "Tracking.cc signal handler")
    if "ConfigureSemanticMasker(strSettingPath);" not in text:
        if "    initID = 0; lastID = 0;\n" in text:
            text = replace_once(text, "    initID = 0; lastID = 0;\n", "    ConfigureSemanticMasker(strSettingPath);\n\n    initID = 0; lastID = 0;\n", "Tracking.cc constructor semantic init")
        else:
            text = replace_once(text, "    initID = 0;\n", "    ConfigureSemanticMasker(strSettingPath);\n\n    initID = 0;\n", "Tracking.cc constructor semantic init")
    if "mpSemanticMasker->Stop()" not in text:
        text = replace_once(
            text,
            "Tracking::~Tracking()\n{\n",
            "Tracking::~Tracking()\n{\n    if(mpSemanticMasker)\n        mpSemanticMasker->Stop();\n",
            "Tracking.cc destructor stop",
        )
    if "unsigned long long Tracking::SubmitSemanticFrame" not in text:
        text = insert_before(text, "void Tracking::newParameterLoader", semantic_methods_block(), "Tracking.cc newParameterLoader")
    if "semanticRequestId" not in text:
        start = text.find("Sophus::SE3f Tracking::GrabImageRGBD")
        end = text.find("Sophus::SE3f Tracking::GrabImageMonocular", start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not isolate Tracking::GrabImageRGBD")
        block = text[start:end]
        block = replace_once(
            block,
            "Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)\n{\n    mImGray = imRGB;\n",
            "Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)\n{\n"
            "    unsigned long long semanticRequestId = 0;\n"
            "    const bool useSemanticMask = ShouldUseSemanticMaskForCurrentFrame();\n"
            "    if(useSemanticMask && mSemanticConfig.submitBeforeOrb)\n"
            "        semanticRequestId = SubmitSemanticFrame(imRGB, timestamp);\n\n"
            "    mImGray = imRGB;\n",
            "Tracking.cc RGBD semantic submit before ORB",
        )
        block = replace_once(
            block,
            "    mCurrentFrame.mNameFile = filename;\n    mCurrentFrame.mnDataset = mnNumDataset;\n\n#ifdef REGISTER_TIMES\n",
            "    if(useSemanticMask && !mSemanticConfig.submitBeforeOrb)\n"
            "        semanticRequestId = SubmitSemanticFrame(imRGB, timestamp);\n"
            "    mCurrentFrame.mNameFile = filename;\n"
            "    mCurrentFrame.mnDataset = mnNumDataset;\n"
            "    if(useSemanticMask)\n"
            "        ApplySemanticMaskForFrame(semanticRequestId);\n\n#ifdef REGISTER_TIMES\n",
            "Tracking.cc RGBD semantic mask before tracking",
        )
        text = text[:start] + block + text[end:]
    write(path, text)


def main():
    if len(sys.argv) != 3:
        print("usage: patch_nitro_with_semantics.py /opt/nitroslam /tmp/dynslam_semantic", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    semantic = Path(sys.argv[2]).resolve()
    patch_cmake(root)
    patch_semantic_files(root, semantic)
    patch_frame(root, semantic)
    patch_tracking_header(root)
    patch_tracking_cc(root, semantic)
    print("Patched Nitro-SLAM with DynSLAM semantic filtering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
