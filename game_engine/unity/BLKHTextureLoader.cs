// SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
// SPDX-License-Identifier: MIT
// Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
// Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole

using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// BLKH Texture Loader for Unity.
/// 
/// Fetches BLKH-compressed textures from a BLKH Texture Streaming Server
/// and loads them as Unity Texture2D at runtime.
/// 
/// Usage:
///   var loader = gameObject.AddComponent<BLKHTextureLoader>();
///   loader.serverUrl = "http://localhost:8080";
///   StartCoroutine(loader.LoadTexture("skybox.blkh8", (texture) => {
///       material.mainTexture = texture;
///   }));
/// 
/// Two modes:
///   1. Server-side decode (default): fetches decoded PNG from server.
///      Simple, works everywhere, but transfers full-size PNG.
///   2. Client-side decode (advanced): fetches raw .blkh8 recipe,
///      requires BLKH native plugin (future work).
/// </summary>
public class BLKHTextureLoader : MonoBehaviour
{
    [Header("BLKH Server")]
    [Tooltip("URL of the BLKH Texture Streaming Server")]
    public string serverUrl = "http://localhost:8080";

    [Header("Settings")]
    [Tooltip("If true, fetches decoded PNG (simple). If false, fetches raw BLKH recipe (needs native decoder).")]
    public bool serverSideDecode = true;

    /// <summary>
    /// Load a BLKH texture by name from the server.
    /// Calls onComplete with the loaded Texture2D, or null on error.
    /// </summary>
    /// <param name="textureName">Name of the .blkh8 file on the server (e.g. "skybox.blkh8")</param>
    /// <param name="onComplete">Callback: (Texture2D texture) — null if failed</param>
    public IEnumerator LoadTexture(string textureName, Action<Texture2D> onComplete)
    {
        string url;
        if (serverSideDecode)
        {
            // Server decodes to PNG — simple path
            url = $"{serverUrl}/texture/{textureName}/decode";
        }
        else
        {
            // Raw recipe bytes — needs native BLKH decoder (future)
            url = $"{serverUrl}/texture/{textureName}";
        }

        using (UnityWebRequest request = UnityWebRequestTexture.GetTexture(url))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Texture2D texture = DownloadHandlerTexture.GetContent(request);
                texture.name = textureName;
                Debug.Log($"[BLKH] Loaded texture '{textureName}': {texture.width}x{texture.height} " +
                          $"(server-side decode: {serverSideDecode})");
                onComplete?.Invoke(texture);
            }
            else
            {
                Debug.LogError($"[BLKH] Failed to load '{textureName}': {request.error}");
                onComplete?.Invoke(null);
            }
        }
    }

    /// <summary>
    /// Load multiple BLKH textures in parallel.
    /// Calls onComplete when ALL textures are loaded.
    /// </summary>
    public IEnumerator LoadTextures(string[] textureNames, Action<Texture2D[]> onComplete)
    {
        Texture2D[] textures = new Texture2D[textureNames.Length];
        int loadedCount = 0;

        for (int i = 0; i < textureNames.Length; i++)
        {
            int index = i; // Capture for closure
            StartCoroutine(LoadTexture(textureNames[i], (tex) =>
            {
                textures[index] = tex;
                loadedCount++;
                if (loadedCount == textureNames.Length)
                {
                    onComplete?.Invoke(textures);
                }
            }));
        }

        yield return null;
    }

    /// <summary>
    /// Get metadata about a texture without downloading it.
    /// </summary>
    public IEnumerator GetTextureInfo(string textureName, Action<string> onComplete)
    {
        string url = $"{serverUrl}/texture/{textureName}/info";

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                onComplete?.Invoke(request.downloadHandler.text);
            }
            else
            {
                Debug.LogError($"[BLKH] Failed to get info for '{textureName}': {request.error}");
                onComplete?.Invoke(null);
            }
        }
    }

    /// <summary>
    /// Compress a PNG texture on the server and save it as BLKH.
    /// Useful for in-editor tools: compress once, stream forever.
    /// </summary>
    public IEnumerator CompressTexture(byte[] pngData, string outputName, Action<string> onComplete)
    {
        string url = $"{serverUrl}/compress?name={outputName}";

        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(pngData);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "image/png");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"[BLKH] Compressed '{outputName}': {request.downloadHandler.text}");
                onComplete?.Invoke(request.downloadHandler.text);
            }
            else
            {
                Debug.LogError($"[BLKH] Compress failed: {request.error}");
                onComplete?.Invoke(null);
            }
        }
    }

    /// <summary>
    /// List all available textures on the server.
    /// </summary>
    public IEnumerator ListTextures(Action<string> onComplete)
    {
        string url = $"{serverUrl}/textures";

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                onComplete?.Invoke(request.downloadHandler.text);
            }
            else
            {
                onComplete?.Invoke(null);
            }
        }
    }
}

/// <summary>
/// Example usage: load a skybox texture on startup.
/// Attach this to a GameObject with a Material.
/// </summary>
public class BLKHExample : MonoBehaviour
{
    public string textureName = "skybox.blkh8";
    public string serverUrl = "http://localhost:8080";
    private Renderer rend;

    void Start()
    {
        rend = GetComponent<Renderer>();
        var loader = gameObject.AddComponent<BLKHTextureLoader>();
        loader.serverUrl = serverUrl;

        StartCoroutine(loader.LoadTexture(textureName, (texture) =>
        {
            if (texture != null && rend != null)
            {
                rend.material.mainTexture = texture;
                Debug.Log($"[BLKH] Skybox loaded: {textureName}");
            }
        }));
    }
}
