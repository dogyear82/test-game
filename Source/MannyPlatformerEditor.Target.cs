using UnrealBuildTool;
using System.Collections.Generic;

public class MannyPlatformerEditorTarget : TargetRules
{
    public MannyPlatformerEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_2;
        ExtraModuleNames.AddRange(new string[] { "MannyPlatformer" });
    }
}

