#include <linux/build-salt.h>
#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(.gnu.linkonce.this_module) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section(__versions) = {
	{ 0x64491f21, "module_layout" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0x2d5589da, "__register_chrdev" },
	{ 0x5b3f3e79, "nvidia_p2p_get_pages" },
	{ 0xdecd0b29, "__stack_chk_fail" },
	{ 0x642487ac, "nvidia_p2p_put_pages" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0xedc03953, "iounmap" },
	{ 0xc5850110, "printk" },
	{ 0x93a219c, "ioremap_nocache" },
	{ 0xf42ca687, "nvidia_p2p_free_page_table" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "nvidia");


MODULE_INFO(srcversion, "85B9282C0EA156FF5D6C514");
