using UnrealBuildTool;
using System.Collections.Generic;

public class MannyPlatformerTarget : TargetRules
{
    public MannyPlatformerTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V5;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_2;
        ExtraModuleNames.AddRange(new string[] { "MannyPlatformer" });
    }
}

