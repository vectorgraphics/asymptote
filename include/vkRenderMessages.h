#pragma once

enum VulkanRendererMessage : uint32_t
{
  /** Export, if ready */
  exportRender = 0,

  /** Update the renderer */
  updateRenderer = 1
};